import sys
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
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

# Load environment variables
load_dotenv()

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
    global vector_store

    if not vector_store:
        vector_store = load_vector_store()
        if not vector_store:
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

    try:
        # Use k=3 per subquery as requested
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 2. Run retrieval for each subquery and collect unique chunks
        seen_keys: set[tuple] = set()
        unique_docs = []

        for sq in subqueries:
            logger.info("RAG internal retrieval query: %s", sq)
            docs = retriever.invoke(sq)

            if not docs:
                logger.info("RAG retrieved 0 documents for subquery: %s", sq)
                continue

            logger.info("RAG retrieved %d documents for subquery '%s'", len(docs), sq)
            for d in docs:
                key = (
                    d.metadata.get("source_file", ""),
                    d.metadata.get("chunk_id", ""),
                    d.page_content,
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_docs.append(d)

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
            return "No relevant entities found in the knowledge graph."

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
                        "  MATCH (a:Concept {id:$a})<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(b:Concept {id:$b}) RETURN d LIMIT 10\n"
                        "  or shortestPath((a)-[*..4]-(b))\n"
                        "- Use the grounded entity ids literally."
                    )
            except Exception:
                pass

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

            if len(steps) > 1 and "context" in steps[1]:
                raw_context_rows = steps[1]["context"]
                logger.info(
                    "Raw graph context rows returned from Neo4j:\n%s",
                    raw_context_rows,
                )

        final_graph_context = result.get("result", "No result returned.")

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

                # (B) Co-mention evidence through documents (very common in your schema)
                comention = graph.query(
                    """
                    MATCH (a:Concept {id:$a})<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(b:Concept {id:$b})
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

        logger.info("Graph context returned to LLM:\n%s", final_graph_context)

        return final_graph_context

    except Exception as e:
        logger.error(f"Error in Graph retrieval: {e}", exc_info=True)
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

    # 1) Always call RAG tool
    logger.info("Invoking RAG retrieval tool...")
    rag_result = rag_retrieval_tool.invoke(args.query)

    # 2) Always call Graph tool
    logger.info("Invoking Graph retrieval tool...")
    graph_result = graph_retrieval_tool.invoke(args.query)

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
Provide a clear, concise answer to the user's question.
"""

    logger.info("Synthesizing final answer from RAG and Graph contexts...")
    try:
        final_msg = llm_for_tools.invoke(synthesis_prompt)
        final_text = (
            final_msg.content if isinstance(final_msg, str) else final_msg.content
        )
        logger.info("Final Answer: %s", final_text)
        print("\n=== Final Answer ===\n")
        print(final_text)
    except Exception as e:
        logger.exception("Final answer synthesis failed.")
        print("Error during final answer synthesis:", e)


if __name__ == "__main__":
    main()
