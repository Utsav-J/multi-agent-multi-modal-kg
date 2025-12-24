import sys
import os
import argparse
import logging
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from langchain_community.vectorstores import FAISS
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

from utils.rag_rephrase import generate_rag_subqueries

vector_store = None
llm_for_tools = None
LAST_QUERY_TRACE: dict = {
    "rag": None,
    "graph": None,
}

# Load environment variables
load_dotenv()

# Windows console often defaults to cp1252 which can crash logging when unicode appears
# in retrieved chunks/markdown. Force UTF-8 best-effort.
try:
    
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Configure logging: log to both console and a file under `logs/query_agent_logs.txt`
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "query_agent_logs.txt"
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[stream_handler, file_handler],
)
logger = logging.getLogger(__name__)

# Define paths
VECTOR_STORE_DIR = project_root / "vector_store_outputs"
CHUNKING_OUTPUTS_DIR = project_root / "chunking_outputs"


class EmbeddingGemmaWrapper(Embeddings):
    """Wrapper for Google's EmbeddingGemma model via SentenceTransformers."""

    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        embeddings = self.model.encode(
            texts,
            prompt_name="document",
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        embedding = self.model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
        )
        return embedding.tolist()


def load_vector_store():
    """Loads the existing FAISS vector store from disk."""
    try:
        index_path = VECTOR_STORE_DIR / "index"
        if not index_path.exists():
            logger.error(f"Vector store not found at {index_path}")
            return None

        logger.info("Loading EmbeddingGemma model...")
        embeddings = EmbeddingGemmaWrapper(model_name="google/embeddinggemma-300m")

        logger.info(f"Loading vector store from {index_path}...")
        vector_store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None


CYPHER_PROMPT = PromptTemplate(
    input_variables=[
        "schema",
        "query",
        "entities",
        "relationship_types",
        "cypher_hints",
    ],
    template="""
You are an expert Neo4j Cypher generator.

Schema:
{schema}

Known entities in the graph (use ONLY these names):
{entities}

Allowed relationship types (use ONLY these if you specify a relationship type):
{relationship_types}

Query hints (follow these):
{cypher_hints}

Rules:
- Do NOT invent entity names.
- Use only labels and properties present in the schema.
- Do NOT invent relationship types. If you are unsure which relationship type applies, use an untyped pattern like: (a)-[r]-(b)
- If no entity matches, return an empty query.
- Output Cypher only (no prose, no "cypher" prefix).

User question:
{query}

Generate Cypher only.
""",
)


def resolve_entities(graph: Neo4jGraph, query: str, limit: int = 5):
    """Resolve entity candidates from Neo4j using a full-text index."""
    cypher = """
    CALL db.index.fulltext.queryNodes(
        "entityIndex",
        $query
    )
    YIELD node, score
    RETURN
        labels(node) AS labels,
        coalesce(node.id, node.text, node.name) AS name,
        score
    ORDER BY score DESC
    LIMIT $limit
    """

    rows = graph.query(
        cypher,
        params={"query": query, "limit": limit},
    )
    # Filter out rows where the "name" is missing; these are unusable for grounding.
    return [r for r in (rows or []) if r.get("name")]


@tool
def rag_retrieval_tool(query: str) -> str:
    """
    Performs RAG retrieval on the document vector store.
    Useful for retrieving specific text chunks or context from the processed documents.

    Args:
        query (str): The search query.

    Returns:
        str: Retrieved documents concatenated.
    """
    logger.info("RAG Tool invoked with user query: %s", query)
    global vector_store, LAST_QUERY_TRACE

    rag_trace: dict = {
        "query": query,
        "rewritten_or_decomposed_queries": [],
        "k_per_subquery": 3,
        "retrieved_chunks": [],
        "retrieved_by_subquery": [],
        "error": None,
    }
    t0 = time.perf_counter()

    if not vector_store:
        vector_store = load_vector_store()
        if not vector_store:
            rag_trace["error"] = "Vector store is not available."
            rag_trace["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            LAST_QUERY_TRACE["rag"] = rag_trace
            return "Error: Vector store is not available."

    # 1. Generate optimized RAG subqueries using Groq-backed util
    try:
        subqueries = generate_rag_subqueries(query)
    except Exception as e:
        logger.error(
            "Failed to generate RAG subqueries, falling back to original query: %s", e
        )
        subqueries = [query]

    if not subqueries:
        subqueries = [query]

    logger.info("Generated %d RAG subqueries:", len(subqueries))
    for i, sq in enumerate(subqueries, 1):
        logger.info("  Subquery %d: %s", i, sq)

    rag_trace["rewritten_or_decomposed_queries"] = list(subqueries)

    try:
        # Use k=3 per subquery as requested
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 2. Run retrieval for each subquery and collect unique chunks
        seen_keys: set[tuple] = set()
        unique_docs = []
        retrieved_by_subquery: list[dict] = []

        for sq in subqueries:
            logger.info("RAG internal retrieval query: %s", sq)
            docs = retriever.invoke(sq)

            if not docs:
                logger.info("RAG retrieved 0 documents for subquery: %s", sq)
                continue

            logger.info("RAG retrieved %d documents for subquery '%s'", len(docs), sq)
            subquery_hits: list[dict] = []
            for d in docs:
                key = (
                    d.metadata.get("source_file", ""),
                    d.metadata.get("chunk_id", ""),
                    d.page_content,
                )
                subquery_hits.append(
                    {
                        "chunk_id": d.metadata.get("chunk_id", "unknown"),
                        "source_file": d.metadata.get("source_file", "unknown"),
                    }
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_docs.append(d)

            retrieved_by_subquery.append({"subquery": sq, "hits": subquery_hits})

        # 3. Log each unique retrieved document with clear differentiation and metadata (no content)
        logger.info(
            "Total unique documents after de-duplication across all subqueries: %d",
            len(unique_docs),
        )
        if unique_docs:
            for i, d in enumerate(unique_docs, start=1):
                logger.info(
                    "RAG Unique Doc %d:\n  Source: %s\n  Chunk ID: %s\n  Metadata: %s",
                    i,
                    d.metadata.get("source_file", "unknown"),
                    d.metadata.get("chunk_id", "unknown"),
                    {
                        k: v
                        for k, v in d.metadata.items()
                        if k not in {"source_file", "chunk_id"}
                    },
                )
        else:
            logger.info("RAG retrieved 0 unique documents across all subqueries.")

        # 4. Build result text over unique_docs
        result_text = "\n\n".join(
            [
                f"Content: {d.page_content}\nSource: {d.metadata.get('source_file', 'unknown')}"
                for d in unique_docs
            ]
        )

        rag_trace["retrieved_by_subquery"] = retrieved_by_subquery
        rag_trace["retrieved_chunks"] = [
            {
                "chunk_id": d.metadata.get("chunk_id", "unknown"),
                "source_file": d.metadata.get("source_file", "unknown"),
                "metadata": dict(d.metadata or {}),
                "text": d.page_content,
            }
            for d in unique_docs
        ]
        rag_trace["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        LAST_QUERY_TRACE["rag"] = rag_trace

        # Log only metadata summary of the context returned to the LLM, not full content
        logger.info(
            "RAG context returned to LLM (multi-subquery, k=3, deduplicated): "
            "num_chunks=%d, total_chars=%d",
            len(unique_docs),
            len(result_text),
        )

        return result_text if result_text else "No relevant documents found."

    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}", exc_info=True)
        rag_trace["error"] = str(e)
        rag_trace["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        LAST_QUERY_TRACE["rag"] = rag_trace
        return f"Error occurred during retrieval: {str(e)}"


@tool
def graph_retrieval_tool(query: str) -> str:
    """
    Queries the knowledge graph database (Neo4j) using Cypher.
    Uses entity grounding + a constrained Cypher generation prompt to avoid hallucinated entities.

    Args:
        query (str): The natural language query about the graph.

    Returns:
        str: The answer derived from the graph database.
    """
    logger.info(f"Graph Tool invoked with user query: {query}")
    global LAST_QUERY_TRACE
    t0 = time.perf_counter()

    graph_trace: dict = {
        "query": query,
        "grounded_entities": [],
        "generated_cypher": None,
        "raw_context_rows": None,
        "fallback_used": False,
        "retrieved_subgraph": {"nodes": [], "edges": []},
        "error": None,
    }

    try:
        url = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        logger.info(f"Connecting to Neo4j at {url}...")
        graph = Neo4jGraph(url=url, username=username, password=password)

        schema = graph.schema
        # Relationship types from Neo4j (more robust than trying to parse schema text)
        try:
            rel_rows = graph.query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
            )
            relationship_types = ", ".join(
                [
                    r.get("relationshipType")
                    for r in (rel_rows or [])
                    if r.get("relationshipType")
                ]
            )
        except Exception:
            relationship_types = ""

        # Use the shared LLM if available, otherwise create a local one
        llm = llm_for_tools or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        )

        # 1. Resolve entities FIRST
        logger.info("Resolving entities from graph for query: %s", query)
        entities = resolve_entities(graph, query)

        if not entities:
            logger.warning("No matching entities found.")
            graph_trace["grounded_entities"] = []
            graph_trace["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            LAST_QUERY_TRACE["graph"] = graph_trace
            return "No relevant entities found in the knowledge graph."

        graph_trace["grounded_entities"] = entities

        entity_str = "\n".join(
            f"- {e['name']} ({', '.join(e['labels'])})" for e in entities
        )

        logger.info(
            "Grounded entities to be provided to Cypher generator:\n%s", entity_str
        )

        # Provide hints to avoid brittle "semantic edge guessing". If there is no
        # direct Concept-Concept edge between the top two concepts, prefer co-mention
        # via Document or shortestPath queries that reflect how this KG is structured.
        def _top2_concepts_for_hints(resolved_entities: list[dict]) -> list[str]:
            out: list[str] = []
            for e in resolved_entities:
                if e.get("name") and "Concept" in (e.get("labels") or []):
                    out.append(e["name"])
            return out[:2]

        cypher_hints = "- If uncertain, use simple MATCH patterns and LIMIT results."
        top2_for_hints = _top2_concepts_for_hints(entities)
        if len(top2_for_hints) == 2:
            a, b = top2_for_hints
            try:
                direct_count = graph.query(
                    """
                    MATCH (a:Concept {id:$a})-[r]-(b:Concept {id:$b})
                    RETURN count(r) AS c
                    """,
                    params={"a": a, "b": b},
                )
                c = (direct_count or [{}])[0].get("c", 0)
                if not c:
                    cypher_hints = (
                        f"- There may be NO direct edge between Concept '{a}' and Concept '{b}'.\n"
                        "- Prefer evidence queries like:\n"
                        "  MATCH (a:Concept {id:$a})<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(b:Concept {id:$b})\n"
                        "  WHERE d.source_type = 'chunk'\n"
                        "  RETURN d.source_id, d.chunk_file, d.chunk_id, d.chunk_index LIMIT 10\n"
                        "  or shortestPath((a)-[*..4]-(b))\n"
                        "- Use the grounded entity ids literally."
                    )
            except Exception:
                pass

        # Image evidence helper: pull Image nodes + their paths connected to grounded concepts.
        def _extract_image_paths_for_concepts(
            concept_ids: list[str], limit: int = 10
        ) -> list[dict]:
            if not concept_ids:
                return []
            rows = graph.query(
                """
                UNWIND $concept_ids AS cid
                MATCH (c:Concept {id: cid})
                OPTIONAL MATCH (i1:Image)-[:DEPICTS]->(c)
                OPTIONAL MATCH (c)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(i2:Image)
                WITH cid,
                     collect(DISTINCT i1) + collect(DISTINCT i2) AS imgs
                UNWIND imgs AS i
                WITH cid, i
                WHERE i IS NOT NULL
                WITH cid, i, properties(i) AS p
                RETURN DISTINCT
                  i.id AS image_id,
                  coalesce(p.source_path, p.path) AS image_path,
                  cid AS related_concept
                LIMIT $limit
                """,
                params={"concept_ids": concept_ids, "limit": limit},
            )
            # Keep only rows with a path
            return [
                r for r in (rows or []) if r.get("image_id") and r.get("image_path")
            ]

        # Chunk evidence helpers: resolve Document.source_id -> chunk in chunking_outputs/<chunk_filename>
        def _extract_strings_deep(obj):
            """Yield all string values from nested dict/list structures."""
            stack = [obj]
            while stack:
                cur = stack.pop()
                if cur is None:
                    continue
                if isinstance(cur, str):
                    yield cur
                elif isinstance(cur, dict):
                    for v in cur.values():
                        stack.append(v)
                elif isinstance(cur, (list, tuple)):
                    for v in cur:
                        stack.append(v)

        def _extract_source_ids_from_rows(rows: list[dict] | None) -> list[str]:
            if not rows:
                return []
            out: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for s in _extract_strings_deep(row):
                    # Resolve only chunk JSONLs (skip markdown pointers like *.md::images)
                    if ".jsonl::" in s and s not in seen:
                        seen.add(s)
                        out.append(s)
            return out

        def _parse_chunk_source_id(source_id: str) -> tuple[str, str] | None:
            """
            Expected format:
              <chunk_filename>::<chunk_entry_id>
            Example:
              attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl::attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k_1
            """
            if not isinstance(source_id, str) or "::" not in source_id:
                return None
            chunk_filename, chunk_entry_id = source_id.split("::", 1)
            if not chunk_filename.endswith(".jsonl") or not chunk_entry_id:
                return None
            return chunk_filename, chunk_entry_id

        def _load_chunk_entry(chunk_filename: str, chunk_entry_id: str) -> dict | None:
            chunk_path = CHUNKING_OUTPUTS_DIR / chunk_filename
            if not chunk_path.exists():
                return None
            try:
                import json

                with open(chunk_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if obj.get("id") == chunk_entry_id:
                            return obj
            except Exception:
                return None
            return None

        def _extract_chunk_filenames_from_rows(rows: list[dict] | None) -> list[str]:
            """
            Extract chunk JSONL filenames from returned rows (e.g., derived_from_chunk_file, chunk_file).
            """
            if not rows:
                return []
            out: list[str] = []
            seen: set[str] = set()
            for row in rows:
                # Fast path for common keys
                for k in ("derived_from_chunk_file", "chunk_file", "d.chunk_file"):
                    v = row.get(k)
                    if isinstance(v, str) and v.endswith(".jsonl") and v not in seen:
                        seen.add(v)
                        out.append(v)
                # Deep scan for any *.jsonl filenames
                for s in _extract_strings_deep(row):
                    if s.endswith(".jsonl") and s not in seen:
                        seen.add(s)
                        out.append(s)
            return out

        def _find_chunks_in_file_by_terms(
            chunk_filename: str, terms: list[str], limit: int = 5
        ) -> list[dict]:
            """
            Best-effort: scan a chunk file and return chunk entries whose content mentions any term.
            Used when the graph returns a markdown Document that only points back to a chunk file
            (e.g., derived_from_chunk_file) but not a specific chunk id.
            """
            if not terms:
                return []
            chunk_path = CHUNKING_OUTPUTS_DIR / chunk_filename
            if not chunk_path.exists():
                return []
            lowered_terms = [
                t.lower() for t in terms if isinstance(t, str) and t.strip()
            ]
            if not lowered_terms:
                return []
            matches: list[dict] = []
            try:
                import json

                with open(chunk_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        content = obj.get("content", "")
                        if not isinstance(content, str):
                            continue
                        hay = content.lower()
                        if any(t in hay for t in lowered_terms):
                            matches.append(obj)
                            if len(matches) >= limit:
                                break
            except Exception:
                return []
            return matches

        # 2. Build Cypher chain with constrained prompt
        logger.info("Initializing grounded GraphCypherQAChain...")
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=CYPHER_PROMPT,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

        # 3. Invoke with grounded context
        result = chain.invoke(
            {
                "query": query,
                "entities": entity_str,
                "schema": schema,
                "relationship_types": relationship_types,
                "cypher_hints": cypher_hints,
            }
        )

        # Log intermediate Cypher + raw graph context
        steps = result.get("intermediate_steps", [])
        generated_cypher = None
        raw_context_rows = None
        if steps:
            logger.info("Generated Cypher query for graph retrieval:")
            generated_cypher = steps[0].get("query") or steps[0].get("cypher")
            logger.info(generated_cypher)
            graph_trace["generated_cypher"] = generated_cypher

            if len(steps) > 1 and "context" in steps[1]:
                raw_context_rows = steps[1]["context"]
                logger.info(
                    "Raw graph context rows returned from Neo4j:\n%s",
                    raw_context_rows,
                )
                graph_trace["raw_context_rows"] = raw_context_rows

        final_graph_context = result.get("result", "No result returned.")
        # Keep any fallback query rows that may include Document.source_id values
        fallback_rows_for_sources: list[dict] = []
        fallback_rows_all: list[dict] = []

        # ---------------------------------------------------------------------
        # Robust fallback retrieval
        # ---------------------------------------------------------------------
        # Many KGs (including yours) connect concepts primarily via Document mention
        # edges, so "semantic" relationships may not exist even when entities do.
        # If the LLM-generated Cypher returns no rows, fall back to deterministic
        # graph queries: direct edges, co-mention through documents, and short paths.
        def _pick_top_concepts(resolved_entities: list[dict]) -> list[str]:
            concept_ids: list[str] = []
            for e in resolved_entities:
                labels = e.get("labels") or []
                if "Concept" in labels and e.get("name"):
                    concept_ids.append(e["name"])
            if len(concept_ids) >= 2:
                return concept_ids[:2]
            # Fallback: just use the first two names regardless of label
            names = [e.get("name") for e in resolved_entities if e.get("name")]
            return names[:2]

        def _format_rows(title: str, rows: list[dict]) -> str:
            if not rows:
                return ""
            return f"{title}:\n" + "\n".join([str(r) for r in rows])

        # Prefer returning *raw rows* (evidence) rather than the chain's natural-language
        # summary, because the chain may answer "I don't know" even when rows exist.
        if raw_context_rows:
            final_graph_context = "Raw graph rows:\n" + "\n".join(
                [str(r) for r in raw_context_rows]
            )

        needs_fallback = (not raw_context_rows) and (
            raw_context_rows == []
            or final_graph_context.strip()
            in {"I don't know the answer.", "I don't know the answer"}
        )

        # If the model used invalid relationship types, try sanitizing the Cypher
        # before falling back to heuristic queries.
        if needs_fallback and generated_cypher and relationship_types:
            try:
                import re

                allowed_set = {
                    t.strip()
                    for t in relationship_types.split(",")
                    if isinstance(t, str) and t.strip()
                }

                # Remove accidental "cypher" prefix if present.
                cypher_text = generated_cypher.strip()
                if cypher_text.lower().startswith("cypher"):
                    cypher_text = "\n".join(cypher_text.splitlines()[1:]).strip()

                rel_pat = re.compile(r"\[(?P<inside>[^\]]*?)\:(?P<types>[A-Z0-9_|]+)\]")

                removed: set[str] = set()

                def _rewrite(match: re.Match) -> str:
                    inside = match.group("inside")
                    types = match.group("types")
                    keep = [t for t in types.split("|") if t in allowed_set]
                    drop = [t for t in types.split("|") if t not in allowed_set]
                    removed.update(drop)
                    if not keep:
                        # No valid types -> untyped relationship
                        return f"[{inside}]"
                    return f"[{inside}:{'|'.join(keep)}]"

                sanitized = rel_pat.sub(_rewrite, cypher_text)
                if sanitized != cypher_text:
                    logger.info(
                        "Sanitized invalid relationship types from Cypher (removed=%s). Retrying sanitized query...",
                        sorted(list(removed)),
                    )
                    try:
                        rows = graph.query(sanitized)
                        if rows:
                            final_graph_context = (
                                "Graph context (sanitized Cypher after invalid relationship types):\n"
                                + "\n".join([str(r) for r in rows])
                            )
                            needs_fallback = False
                    except Exception:
                        # If sanitized query fails, we'll continue to heuristic fallbacks
                        pass
            except Exception:
                # Don't let sanitization errors break graph retrieval
                pass

        if needs_fallback:
            logger.info(
                "Graph retrieval returned empty context; running fallback queries..."
            )
            graph_trace["fallback_used"] = True
            top2 = _pick_top_concepts(entities)
            fallback_parts: list[str] = []

            # (A) Direct edges between top concepts (any relationship type)
            if len(top2) == 2:
                a, b = top2
                direct = graph.query(
                    """
                    MATCH (a:Concept {id:$a})-[r]-(b:Concept {id:$b})
                    RETURN a.id AS a, type(r) AS rel, b.id AS b
                    LIMIT 20
                    """,
                    params={"a": a, "b": b},
                )
                fallback_parts.append(
                    _format_rows("Direct Concept-Concept edges", direct)
                )
                if direct:
                    fallback_rows_all.extend(direct)

                # (B) Co-mention evidence through documents (very common in your schema)
                comention = graph.query(
                    """
                    MATCH (a:Concept {id:$a})<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(b:Concept {id:$b})
                    WHERE d.source_type = 'chunk'
                    RETURN
                      d.source_id AS source_id,
                      d.source_type AS source_type,
                      d.chunk_file AS chunk_file,
                      d.chunk_id AS chunk_id,
                      d.chunk_index AS chunk_index,
                      d.markdown_source AS markdown_source
                    LIMIT 10
                    """,
                    params={"a": a, "b": b},
                )
                fallback_parts.append(
                    _format_rows("Co-mention via Document evidence", comention)
                )
                if comention:
                    fallback_rows_for_sources.extend(comention)
                    fallback_rows_all.extend(comention)

                # (C) Short path (any relationships) to surface how nodes connect
                path_rows = graph.query(
                    """
                    MATCH (a:Concept {id:$a})
                    MATCH (b:Concept {id:$b})
                    WITH a, b
                    MATCH p = shortestPath((a)-[*..4]-(b))
                    RETURN p
                    LIMIT 5
                    """,
                    params={"a": a, "b": b},
                )
                fallback_parts.append(
                    _format_rows("Shortest paths (<=4 hops)", path_rows)
                )
                if path_rows:
                    fallback_rows_all.extend(path_rows)

            # (D) Neighborhood expansion for the top grounded concept
            if top2:
                a = top2[0]
                neigh = graph.query(
                    """
                    MATCH (a:Concept {id:$a})-[r]-(n)
                    RETURN type(r) AS rel, labels(n) AS n_labels, n.id AS n_id
                    LIMIT 25
                    """,
                    params={"a": a},
                )
                fallback_parts.append(
                    _format_rows(f"Neighborhood of Concept '{a}'", neigh)
                )
                if neigh:
                    fallback_rows_all.extend(neigh)

            fallback_text = "\n\n".join([p for p in fallback_parts if p])
            if fallback_text.strip():
                final_graph_context = (
                    "Fallback graph context (LLM query returned empty rows):\n\n"
                    + fallback_text
                )
            else:
                final_graph_context = (
                    "No graph connections found (even after fallbacks)."
                )

        # ---------------------------------------------------------------------
        # Image path block (always append when images are found)
        # ---------------------------------------------------------------------
        try:
            concept_ids_for_images = _pick_top_concepts(entities)[:2]
            image_rows = _extract_image_paths_for_concepts(
                concept_ids_for_images, limit=15
            )
            if image_rows:
                lines = []
                for r in image_rows:
                    lines.append(
                        f"- {r.get('image_id')} | {r.get('image_path')} | related_to={r.get('related_concept')}"
                    )
                final_graph_context = (
                    final_graph_context
                    + "\n\nImages (from graph):\n"
                    + "\n".join(lines)
                )
        except Exception:
            # Image block is best-effort; don't break graph retrieval
            pass

        # ---------------------------------------------------------------------
        # Chunk resolution block (from Document.source_id)
        # ---------------------------------------------------------------------
        try:
            combined_rows: list[dict] = []
            if raw_context_rows:
                combined_rows.extend(raw_context_rows)
            if fallback_rows_all:
                combined_rows.extend(fallback_rows_all)

            source_ids = _extract_source_ids_from_rows(combined_rows)
            chunk_lines: list[str] = []

            for sid in source_ids:
                parsed = _parse_chunk_source_id(sid)
                if not parsed:
                    continue
                chunk_filename, chunk_entry_id = parsed
                entry = _load_chunk_entry(chunk_filename, chunk_entry_id)
                chunk_path = CHUNKING_OUTPUTS_DIR / chunk_filename

                if not entry:
                    chunk_lines.append(
                        "\n".join(
                            [
                                f"- {sid}",
                                f"  chunk_file: {chunk_filename}",
                                f"  chunk_path: {chunk_path}",
                                "  status: NOT_FOUND_IN_FILE",
                            ]
                        )
                    )
                    continue

                content = entry.get("content", "")
                if isinstance(content, str) and len(content) > 600:
                    excerpt = content[:600] + "…"
                else:
                    excerpt = content

                chunk_lines.append(
                    "\n".join(
                        [
                            f"- {sid}",
                            f"  chunk_file: {chunk_filename}",
                            f"  chunk_path: {chunk_path}",
                            f"  chunk_id: {entry.get('id')}",
                            f"  chunk_index: {entry.get('chunk_index')}",
                            f"  metadata: {entry.get('metadata')}",
                            f"  token_size_config: {entry.get('token_size_config')}",
                            f"  excerpt: {excerpt}",
                        ]
                    )
                )

            if chunk_lines:
                final_graph_context = (
                    final_graph_context
                    + "\n\nChunks (resolved from Document.source_id):\n"
                    + "\n\n".join(chunk_lines)
                )
            else:
                # If we didn't get explicit chunk source_ids (common when the graph returns
                # a markdown Document), try using derived_from_chunk_file + content matching.
                chunk_files = _extract_chunk_filenames_from_rows(combined_rows)
                # Use grounded entity names as search terms (cap for speed/noise).
                terms = [e.get("name") for e in entities if e.get("name")]
                terms = terms[:6]
                inferred_lines: list[str] = []
                for cf in chunk_files[:3]:
                    hits = _find_chunks_in_file_by_terms(cf, terms, limit=3)
                    for entry in hits:
                        sid = f"{cf}::{entry.get('id')}"
                        content = entry.get("content", "")
                        excerpt = (
                            (content[:600] + "…")
                            if isinstance(content, str) and len(content) > 600
                            else content
                        )
                        inferred_lines.append(
                            "\n".join(
                                [
                                    f"- {sid}",
                                    f"  chunk_file: {cf}",
                                    f"  chunk_path: {CHUNKING_OUTPUTS_DIR / cf}",
                                    f"  chunk_id: {entry.get('id')}",
                                    f"  chunk_index: {entry.get('chunk_index')}",
                                    f"  metadata: {entry.get('metadata')}",
                                    f"  token_size_config: {entry.get('token_size_config')}",
                                    "  resolved_via: derived_from_chunk_file + content_match",
                                    f"  excerpt: {excerpt}",
                                ]
                            )
                        )
                if inferred_lines:
                    final_graph_context = (
                        final_graph_context
                        + "\n\nChunks (inferred from derived_from_chunk_file):\n"
                        + "\n\n".join(inferred_lines)
                    )
        except Exception:
            # Chunk resolution is best-effort; don't break graph retrieval
            pass

        logger.info("Graph context returned to LLM:\n%s", final_graph_context)

        # ---------------------------------------------------------------------
        # Retrieve a concrete subgraph (nodes + edges) around grounded entities
        # ---------------------------------------------------------------------
        try:
            grounded_names = [e.get("name") for e in (entities or []) if e.get("name")][
                :10
            ]
            if grounded_names:
                subgraph_rows = graph.query(
                    """
                    MATCH (n)
                    WHERE coalesce(n.id, n.text, n.name) IN $names
                    OPTIONAL MATCH (n)-[r]-(m)
                    WITH n, r, m
                    LIMIT $limit_rows
                    RETURN
                      collect(DISTINCT {
                        id: coalesce(n.id, n.text, n.name),
                        labels: labels(n),
                        properties: properties(n)
                      }) +
                      collect(DISTINCT CASE
                        WHEN m IS NULL THEN NULL
                        ELSE {
                          id: coalesce(m.id, m.text, m.name),
                          labels: labels(m),
                          properties: properties(m)
                        }
                      END) AS nodes,
                      collect(DISTINCT CASE
                        WHEN r IS NULL THEN NULL
                        ELSE {
                          source: coalesce(startNode(r).id, startNode(r).text, startNode(r).name),
                          type: type(r),
                          target: coalesce(endNode(r).id, endNode(r).text, endNode(r).name),
                          properties: properties(r)
                        }
                      END) AS edges
                    """,
                    params={"names": grounded_names, "limit_rows": 200},
                )
                if subgraph_rows:
                    row0 = (subgraph_rows or [{}])[0] or {}
                    nodes = [n for n in (row0.get("nodes") or []) if n]
                    edges = [e for e in (row0.get("edges") or []) if e]
                    # De-dupe nodes by id, preserve first-seen order.
                    seen_n = set()
                    uniq_nodes = []
                    for n in nodes:
                        nid = n.get("id")
                        if not nid or nid in seen_n:
                            continue
                        seen_n.add(nid)
                        uniq_nodes.append(n)
                    # De-dupe edges by (source,type,target)
                    seen_e = set()
                    uniq_edges = []
                    for e in edges:
                        key = (e.get("source"), e.get("type"), e.get("target"))
                        if key in seen_e:
                            continue
                        seen_e.add(key)
                        uniq_edges.append(e)
                    graph_trace["retrieved_subgraph"] = {
                        "nodes": uniq_nodes,
                        "edges": uniq_edges,
                    }
        except Exception:
            # Subgraph capture is best-effort; do not break answering.
            pass

        graph_trace["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        LAST_QUERY_TRACE["graph"] = graph_trace

        return final_graph_context

    except Exception as e:
        logger.error(f"Error in Graph retrieval: {e}", exc_info=True)
        graph_trace["error"] = str(e)
        graph_trace["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        LAST_QUERY_TRACE["graph"] = graph_trace
        return f"Error occurred during graph query: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG Query Agent")
    parser.add_argument(
        "query", nargs="?", default="what is attention?", help="The query to answer."
    )
    args = parser.parse_args()

    global llm_for_tools, vector_store

    # Initialize LLM for final answer synthesis and tool usage
    llm_for_tools = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True
    )

    # Initialize vector store once
    logger.info("Initializing resources...")
    vector_store = load_vector_store()
    if not vector_store:
        logger.warning("Vector store could not be loaded. RAG tool will fail.")

    # Log the original user query clearly
    logger.info("User query received: %s", args.query)
    logger.info("Starting Query Agent with query: '%s'", args.query)

    total_t0 = time.perf_counter()

    # 1) Always call RAG tool
    logger.info("Invoking RAG retrieval tool...")
    rag_t0 = time.perf_counter()
    rag_result = rag_retrieval_tool.invoke(args.query)
    rag_ms = int((time.perf_counter() - rag_t0) * 1000)

    # 2) Always call Graph tool
    logger.info("Invoking Graph retrieval tool...")
    graph_t0 = time.perf_counter()
    graph_result = graph_retrieval_tool.invoke(args.query)
    graph_ms = int((time.perf_counter() - graph_t0) * 1000)

    # 3) Synthesize final answer using both contexts
    synthesis_prompt = f"""
You are a helpful AI assistant that must answer user questions by combining two sources of information:

1. TEXT CONTEXT (RAG from vector store)
2. GRAPH CONTEXT (Neo4j knowledge graph)

User question:
{args.query}

--- TEXT CONTEXT (from rag_retrieval_tool) ---
{rag_result}

--- GRAPH CONTEXT (from graph_retrieval_tool) ---
{graph_result}

Instructions:
- Use BOTH contexts when forming your answer.
- If the contexts disagree, call out the discrepancy and prefer the graph for relational/structural facts,
  and the text for detailed explanations or empirical results.
- Do NOT ignore either source unless it is clearly irrelevant.
- If information is missing from both sources, say that it's not available instead of hallucinating.
- If the GRAPH CONTEXT contains an "Images (from graph)" section, you MUST include a separate "Image paths" block
  in your final answer listing each image id and its path.
- If the GRAPH CONTEXT contains a "Chunks (resolved from Document.source_id)" OR "Chunks (inferred from derived_from_chunk_file)" section,
  you MUST include a separate "Chunks used" block in your final answer listing each chunk source_id (and the chunk_file + chunk_id if present).
  If the GRAPH CONTEXT does NOT contain either section, do NOT include a "Chunks used" block.
Provide a clear, concise answer to the user's question.
"""

    logger.info("Synthesizing final answer from RAG and Graph contexts...")
    try:
        synth_t0 = time.perf_counter()
        final_msg = llm_for_tools.invoke(synthesis_prompt)
        synth_ms = int((time.perf_counter() - synth_t0) * 1000)
        final_text = (
            final_msg.content if isinstance(final_msg, str) else final_msg.content
        )

        def _deep_find_ints(obj, keys: set[str]) -> dict:
            found: dict = {}
            stack = [obj]
            while stack:
                cur = stack.pop()
                if cur is None:
                    continue
                if isinstance(cur, dict):
                    for k, v in cur.items():
                        if isinstance(k, str) and k in keys and isinstance(v, int):
                            found[k] = v
                        stack.append(v)
                elif isinstance(cur, (list, tuple)):
                    for v in cur:
                        stack.append(v)
            return found

        def _extract_token_usage(msg) -> dict | None:
            """
            Best-effort extraction of token usage from LangChain message metadata.
            Availability depends on the underlying model/provider.
            """
            try:
                meta = {}
                if hasattr(msg, "response_metadata") and isinstance(
                    getattr(msg, "response_metadata"), dict
                ):
                    meta.update(getattr(msg, "response_metadata") or {})
                if hasattr(msg, "additional_kwargs") and isinstance(
                    getattr(msg, "additional_kwargs"), dict
                ):
                    meta.update(getattr(msg, "additional_kwargs") or {})

                token_keys = {
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "prompt_token_count",
                    "candidates_token_count",
                    "completion_token_count",
                }
                found = _deep_find_ints(meta, token_keys)
                return found or None
            except Exception:
                return None

        # -----------------------------------------------------------------
        # Deterministic post-processing (avoid hallucinated "chunks used")
        # -----------------------------------------------------------------
        def _strip_section(text: str, header_starts: tuple[str, ...]) -> str:
            """
            Remove a section that starts with any header in header_starts and
            continues until the next blank line.
            """
            lines = text.splitlines()
            out: list[str] = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if any(line.strip().startswith(h) for h in header_starts):
                    # skip header line + following non-empty lines
                    i += 1
                    while i < len(lines) and lines[i].strip() != "":
                        i += 1
                    # also skip the blank separator if present
                    if i < len(lines) and lines[i].strip() == "":
                        i += 1
                    continue
                out.append(line)
                i += 1
            return "\n".join(out).strip()

        def _extract_image_paths_from_graph_context(
            graph_text: str,
        ) -> list[tuple[str, str]]:
            if "Images (from graph):" not in graph_text:
                return []
            block = graph_text.split("Images (from graph):", 1)[1]
            lines = []
            for ln in block.splitlines():
                if ln.startswith("- "):
                    # "- <id> | <path> | related_to=..."
                    parts = [p.strip() for p in ln[2:].split("|")]
                    if len(parts) >= 2:
                        lines.append((parts[0], parts[1]))
                elif ln.strip() == "":
                    break
            # de-dupe preserving order
            seen = set()
            out = []
            for img_id, img_path in lines:
                key = (img_id, img_path)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
            return out

        def _extract_chunk_source_ids_from_graph_context(graph_text: str) -> list[str]:
            marker = None
            if "Chunks (resolved from Document.source_id):" in graph_text:
                marker = "Chunks (resolved from Document.source_id):"
            elif "Chunks (inferred from derived_from_chunk_file):" in graph_text:
                marker = "Chunks (inferred from derived_from_chunk_file):"
            if not marker:
                return []
            block = graph_text.split(marker, 1)[1]
            out = []
            for ln in block.splitlines():
                if ln.startswith("- "):
                    out.append(ln[2:].strip())
            # de-dupe preserving order
            seen = set()
            uniq = []
            for s in out:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            return uniq

        # Strip "Chunks used" if the graph context doesn't contain resolved chunks.
        if (
            "Chunks (resolved from Document.source_id):" not in graph_result
            and "Chunks (inferred from derived_from_chunk_file):" not in graph_result
        ):
            final_text = _strip_section(
                final_text,
                (
                    "**Chunks used:**",
                    "Chunks used:",
                ),
            )

        # Ensure deterministic image paths section when graph provides images.
        img_pairs = _extract_image_paths_from_graph_context(graph_result)
        if img_pairs and "Image paths" not in final_text:
            final_text = (
                final_text.strip()
                + "\n\nImage paths:\n"
                + "\n".join(
                    [f"- {img_id}: {img_path}" for img_id, img_path in img_pairs]
                )
            )

        # Ensure deterministic chunks section when graph provides resolved chunks.
        chunk_sids = _extract_chunk_source_ids_from_graph_context(graph_result)
        if chunk_sids and "Chunks used" not in final_text:
            final_text = (
                final_text.strip()
                + "\n\nChunks used:\n"
                + "\n".join([f"- {sid}" for sid in chunk_sids])
            )

        logger.info("Final Answer: %s", final_text)
        print("\n=== Final Answer ===\n")
        print(final_text)

        # -----------------------------------------------------------------
        # Structured per-query log (JSON) after the final answer
        # -----------------------------------------------------------------
        try:
            trace_record = {
                "run_id": str(uuid.uuid4()),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "user_question": args.query,
                "rewritten_or_decomposed_queries": (
                    (LAST_QUERY_TRACE.get("rag") or {}).get(
                        "rewritten_or_decomposed_queries", []
                    )
                ),
                "retrieved_chunks": (LAST_QUERY_TRACE.get("rag") or {}).get(
                    "retrieved_chunks", []
                ),
                "retrieved_graph_subgraph": (
                    (LAST_QUERY_TRACE.get("graph") or {}).get(
                        "retrieved_subgraph", {"nodes": [], "edges": []}
                    )
                ),
                "final_answer": final_text,
                "token_usage": _extract_token_usage(final_msg),
                "latency_ms": {
                    "rag": rag_ms,
                    "graph": graph_ms,
                    "synthesis": synth_ms,
                    "total": int((time.perf_counter() - total_t0) * 1000),
                },
                "internal": {
                    "rag": LAST_QUERY_TRACE.get("rag"),
                    "graph": LAST_QUERY_TRACE.get("graph"),
                },
            }

            trace_path = logs_dir / "query_traces.jsonl"
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            logger.info(
                "QUERY_TRACE_JSON %s",
                json.dumps(trace_record, ensure_ascii=False),
            )

            print("\n=== Query Trace (JSON) ===\n")
            print(json.dumps(trace_record, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning("Failed to write structured query trace: %s", e)
    except Exception as e:
        logger.exception("Final answer synthesis failed.")
        print("Error during final answer synthesis:", e)


if __name__ == "__main__":
    main()
