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


def get_graph_chain(llm):
    """Initializes the GraphCypherQAChain for Neo4j."""
    try:
        url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        graph = Neo4jGraph(url=url, username=username, password=password)

        chain = GraphCypherQAChain.from_llm(
            llm=llm, graph=graph, verbose=True, allow_dangerous_requests=True
        )
        return chain
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j connection: {e}")
        return None


# Global resources (initialized lazily or on module load if simple)
# For tools to access them, we can initialize them globally or pass them.
# Given tool definitions must be stateless functions generally, we'll use globals or a class based approach.
# For simplicity with the @tool decorator, we'll assume globals are initialized in main() or check if None.

vector_store = None
graph_chain = None
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
    Useful for finding relationships between entities, structured data, and answering questions about the graph structure.

    Args:
        query (str): The natural language query about the graph.

    Returns:
        str: The answer derived from the graph database.
    """
    logger.info(f"Graph Tool invoked with query: {query}")
    global graph_chain

    if not graph_chain:
        # We need LLM to init the chain if it wasn't init yet.
        # This is a fallback if main didn't set it, though main should.
        if llm_for_tools:
            graph_chain = get_graph_chain(llm_for_tools)

        if not graph_chain:
            return "Error: Graph database connection is not available."

    try:
        result = graph_chain.invoke(query)
        return result.get("result", "No result returned from graph query.")
    except Exception as e:
        logger.error(f"Error in Graph retrieval: {e}")
        return f"Error occurred during graph query: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG Query Agent")
    parser.add_argument(
        "query", nargs="?", default="what is attention?", help="The query to answer."
    )
    args = parser.parse_args()

    global llm_for_tools, vector_store, graph_chain

    # Initialize LLM
    llm_for_tools = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True
    )

    # Initialize resources
    logger.info("Initializing resources...")
    vector_store = load_vector_store()
    graph_chain = get_graph_chain(llm_for_tools)

    if not vector_store:
        logger.warning("Vector store could not be loaded. RAG tool will fail.")

    if not graph_chain:
        logger.warning("Neo4j connection failed. Graph tool will fail.")

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
