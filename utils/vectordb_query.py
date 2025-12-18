import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

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

VECTOR_STORE_DIR = project_root / "vector_store_outputs"


class EmbeddingGemmaWrapper(Embeddings):
    """Wrapper for Google's EmbeddingGemma model via SentenceTransformers."""

    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            prompt_name="document",
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
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
        logger.error(f"Failed to load vector store: {e}", exc_info=True)
        return None


def query_vectordb(query: str, k: int = 5) -> str:
    """
    Queries the FAISS vector database using the EmbeddingGemma embeddings.

    Args:
        query (str): Natural language query.
        k (int): Number of results to retrieve.

    Returns:
        str: Retrieved documents concatenated with simple source info.
    """
    logger.info(f"VectorDB query invoked with query: {query}")
    vector_store = load_vector_store()
    if not vector_store:
        return "Error: Vector store is not available."

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        result_text = "\n\n".join(
            [
                f"Content: {d.page_content}\nSource: {d.metadata.get('source_file', 'unknown')}"
                for d in docs
            ]
        )
        return result_text if result_text else "No relevant documents found."
    except Exception as e:
        logger.error(f"Error during vector DB retrieval: {e}", exc_info=True)
        return f"Error occurred during vector DB query: {str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query FAISS Vector DB")
    parser.add_argument(
        "query",
        nargs="?",
        default="what is attention mechanism?",
        help="The query to answer from the vector DB.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5).",
    )
    args = parser.parse_args()

    response = query_vectordb(args.query, k=args.k)
    print("\nVectorDB Results:")
    print(response)
