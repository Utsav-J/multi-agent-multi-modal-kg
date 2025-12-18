import sys
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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


# ----- Graph querying helpers (ported from utils/neo4j_query.py) -----

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


# Global resources (initialized lazily)
vector_store = None
llm_for_tools = None


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
    logger.info(f"RAG Tool invoked with query: {query}")
    global vector_store

    if not vector_store:
        vector_store = load_vector_store()
        if not vector_store:
            return "Error: Vector store is not available."

    try:
        # Simple retrieval for now, can be enhanced to MultiQueryRetriever as planned
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)

        result_text = "\n\n".join(
            [
                f"Content: {d.page_content}\nSource: {d.metadata.get('source_file', 'unknown')}"
                for d in docs
            ]
        )
        return result_text if result_text else "No relevant documents found."

    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}")
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
    logger.info(f"Graph Tool invoked with query: {query}")

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
        logger.info("Resolving entities from graph...")
        entities = resolve_entities(graph, query)

        if not entities:
            logger.warning("No matching entities found.")
            return "No relevant entities found in the knowledge graph."

        entity_str = "\n".join(
            f"- {e['name']} ({', '.join(e['labels'])})" for e in entities
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

        steps = result.get("intermediate_steps", [])
        if steps:
            logger.info("Generated Cypher:")
            # steps[0] is usually {'query': <cypher>} or similar; guard with .get
            logger.info(steps[0].get("query") or steps[0].get("cypher"))

        return result.get("result", "No result returned.")

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

    # Initialize LLM

    # Initialize resources
    logger.info("Initializing resources...")
    vector_store = load_vector_store()

    if not vector_store:
        logger.warning("Vector store could not be loaded. RAG tool will fail.")

    llm_for_tools = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True
    )
    tools = [rag_retrieval_tool, graph_retrieval_tool]

    sys_prompt = (
        "You are a helpful AI assistant capable of answering complex questions by combining information "
        "from text documents (via RAG) and a knowledge graph (via Neo4j). "
        "For a given user query:\n"
        "1. Use 'rag_retrieval_tool' to get relevant text context.\n"
        "2. Use 'graph_retrieval_tool' to understand relationships and structured data.\n"
        "3. Synthesize the information from both sources to provide a comprehensive answer.\n"
        "Always cite your sources if possible (e.g. 'according to the text...' or 'the graph shows...')."
    )

    agent = create_agent(model=llm_for_tools, tools=tools, system_prompt=sys_prompt)

    logger.info(f"Starting Query Agent with query: '{args.query}'")

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
        logger.info(f"Final Answer: {result['messages'][-1].content}")

    except Exception as e:
        logger.exception("Agent execution failed.")


if __name__ == "__main__":
    main()
