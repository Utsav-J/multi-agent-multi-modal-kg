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
    input_variables=["schema", "query", "entities"],
    template="""
You are an expert Neo4j Cypher generator.

Schema:
{schema}

Known entities in the graph (use ONLY these names):
{entities}

Rules:
- Do NOT invent entity names.
- Use only labels and properties present in the schema.
- If no entity matches, return an empty query.

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
        node.text AS name,
        score
    ORDER BY score DESC
    LIMIT $limit
    """

    return graph.query(
        cypher,
        params={"query": query, "limit": limit},
    )


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
            }
        )

        # Log intermediate Cypher + raw graph context
        steps = result.get("intermediate_steps", [])
        if steps:
            logger.info("Generated Cypher query for graph retrieval:")
            logger.info(steps[0].get("query") or steps[0].get("cypher"))

            if len(steps) > 1 and "context" in steps[1]:
                logger.info(
                    "Raw graph context rows returned from Neo4j:\n%s",
                    steps[1]["context"],
                )

        final_graph_context = result.get("result", "No result returned.")
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
