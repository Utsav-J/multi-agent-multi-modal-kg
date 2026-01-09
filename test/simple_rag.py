"""
Simple RAG script for comparison with hybrid RAG+Graph system.

This script performs traditional RAG:
1. Loads FAISS vector store
2. Retrieves relevant chunks for a query
3. Generates an answer using the retrieved context

Can also generate test cases for DeepEval evaluation:
- Interactive mode: Process a single query
- Batch mode: Process all questions from questions.csv

Usage:
    # Simple query
    python test/simple_rag.py "your query here"

    # Generate test case for single query
    python test/simple_rag.py --mode interactive --query "your query"

    # Batch process questions.csv
    python test/simple_rag.py --mode batch --csv-path questions.csv
"""

import sys
import os
import argparse
import logging
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
test_dir = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(test_dir))

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# Import CSV utilities
from csv_questions_utils import load_questions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
VECTOR_STORE_DIR = project_root / "vector_store_outputs"

# Global variables
vector_store = None
llm = None


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
        logger.info("Vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}", exc_info=True)
        return None


def retrieve_chunks(vector_store, query: str, k: int = 1):
    """
    Retrieve relevant chunks from the vector store.

    Args:
        vector_store: The FAISS vector store
        query: The search query
        k: Number of chunks to retrieve

    Returns:
        List of retrieved document chunks
    """
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} chunks for query: {query}")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}", exc_info=True)
        return []


def format_context(docs, max_chars_per_chunk: int = 500):
    """
    Format retrieved documents into context string.

    Args:
        docs: List of retrieved document chunks
        max_chars_per_chunk: Maximum characters to include per chunk (truncates if longer)

    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        content = doc.page_content
        # Truncate content to limit context quality
        if len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk] + "..."

        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content: {content}\n"
        )

    return "\n---\n\n".join(context_parts)


def generate_answer(query: str, context: str, llm_instance):
    """
    Generate an answer using the LLM with the retrieved context.

    Args:
        query: The user's query
        context: The retrieved context
        llm_instance: The language model

    Returns:
        Tuple of (answer_text, token_usage_dict)
    """
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

User question:
{query}

Context from retrieved documents:
{context}

Instructions:
- Answer the question based ONLY on the provided context.
- If the context doesn't contain enough information to answer the question, say so clearly.
- Do NOT make up information that isn't in the context.
- Be concise and accurate.

Answer:"""

    try:
        logger.info("Generating answer using LLM...")
        response = llm_instance.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Extract token usage if available
        token_usage = None
        try:
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                token_usage = {
                    k: v
                    for k, v in meta.items()
                    if "token" in k.lower() and isinstance(v, (int, float))
                }
        except Exception:
            pass

        return answer, token_usage
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return f"Error generating answer: {str(e)}", None


def process_single_query(
    query: str, category: Optional[str] = None, k: int = 1
) -> Dict[str, any]:
    """
    Process a single query through traditional RAG and return all relevant data.

    Args:
        query: The question to process
        category: Optional category name for the question
        k: Number of chunks to retrieve

    Returns:
        Dictionary containing query, retrieval context, final answer, and metadata
    """
    global vector_store, llm

    logger.info(f"Processing query: {query}")

    # Initialize resources if needed
    if vector_store is None:
        logger.info("Loading vector store...")
        vector_store = load_vector_store()
        if not vector_store:
            return {
                "query": query,
                "category": category or "",
                "retrieval_context": "",
                "final_answer": "Error: Vector store not available.",
                "retrieval_latency_ms": 0,
                "generation_latency_ms": 0,
                "total_latency_ms": 0,
                "token_usage": "",
                "rag_chunks_count": 0,
                "rag_error": "Vector store not available",
                "timestamp": datetime.now().isoformat(),
            }

    if llm is None:
        logger.info("Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.8,
            convert_system_message_to_human=True,
        )

    total_t0 = time.perf_counter()

    # 1) Retrieve chunks
    logger.info("Retrieving chunks...")
    retrieval_t0 = time.perf_counter()
    try:
        docs = retrieve_chunks(vector_store, query, k=k)
        retrieval_ms = int((time.perf_counter() - retrieval_t0) * 1000)
        rag_error = None
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        docs = []
        retrieval_ms = 0
        rag_error = str(e)

    # 2) Format context
    context = (
        format_context(docs, max_chars_per_chunk=500)
        if docs
        else "No relevant documents found."
    )

    # 3) Generate answer
    logger.info("Generating answer...")
    generation_t0 = time.perf_counter()
    try:
        final_answer, token_usage = generate_answer(query, context, llm)
        generation_ms = int((time.perf_counter() - generation_t0) * 1000)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        final_answer = f"Error: {str(e)}"
        generation_ms = 0
        token_usage = None

    total_ms = int((time.perf_counter() - total_t0) * 1000)

    # Build result dictionary
    result = {
        "query": query,
        "category": category or "",
        "retrieval_context": context,
        "final_answer": final_answer,
        "retrieval_latency_ms": retrieval_ms,
        "generation_latency_ms": generation_ms,
        "total_latency_ms": total_ms,
        "token_usage": str(token_usage) if token_usage else "",
        "rag_chunks_count": len(docs),
        "rag_error": rag_error or "",
        "timestamp": datetime.now().isoformat(),
    }

    return result


def save_results_to_csv(results: List[Dict[str, any]], output_path: Path):
    """
    Save results to a CSV file compatible with DeepEval.

    Args:
        results: List of result dictionaries
        output_path: Path to save the CSV file
    """
    if not results:
        logger.warning("No results to save.")
        return

    # Define CSV columns (compatible with DeepEval format)
    fieldnames = [
        "query",
        "category",
        "retrieval_context",
        "final_answer",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "token_usage",
        "rag_chunks_count",
        "rag_error",
        "timestamp",
    ]

    # Write to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved {len(results)} results to {output_path}")


def interactive_mode():
    """Interactive mode: Process a single question from user input."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - Single Query Test (Traditional RAG)")
    print("=" * 80)

    query = input("\nEnter your question: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    print(f"\nProcessing query: {query}")
    print("-" * 80)

    result = process_single_query(query)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nQuery: {result['query']}")
    print(
        f"\nRetrieval Context (first 500 chars):\n{result['retrieval_context'][:500]}..."
    )
    print(f"\nFinal Answer:\n{result['final_answer']}")
    print(
        f"\nLatencies: Retrieval={result['retrieval_latency_ms']}ms, "
        f"Generation={result['generation_latency_ms']}ms, "
        f"Total={result['total_latency_ms']}ms"
    )
    print(f"Chunks retrieved: {result['rag_chunks_count']}")

    # Ask if user wants to save
    save = input("\nSave to CSV? (y/n): ").strip().lower()
    if save == "y":
        output_path = (
            project_root / "test" / "outputs" / "simple_rag_test_case_single.csv"
        )
        save_results_to_csv([result], output_path)
        print(f"Saved to {output_path}")


def batch_mode(csv_path: str, output_path: Optional[str] = None, k: int = 1):
    """
    Batch mode: Process all questions from questions.csv file.

    Args:
        csv_path: Path to the questions.csv file
        output_path: Optional custom output path for the results CSV
        k: Number of chunks to retrieve per query
    """
    print("\n" + "=" * 80)
    print("BATCH MODE - Processing Questions from CSV (Traditional RAG)")
    print("=" * 80)

    # Load questions
    questions_by_category = load_questions(csv_path)
    if not questions_by_category:
        logger.error(f"Failed to load questions from {csv_path}")
        return

    sorted_categories = sorted(questions_by_category.keys())
    total_questions = sum(
        len(questions) for questions in questions_by_category.values()
    )

    print(
        f"\nLoaded {total_questions} questions across {len(sorted_categories)} categories"
    )
    print(f"Categories: {', '.join(sorted_categories)}")
    print(f"Retrieval k: {k}")

    # Confirm before processing
    confirm = input("\nProceed with processing all questions? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    # Process questions category by category
    all_results = []
    processed = 0

    for category in sorted_categories:
        questions = questions_by_category[category]
        print(f"\n{'=' * 80}")
        print(f"Processing Category: {category} ({len(questions)} questions)")
        print(f"{'=' * 80}")

        for idx, question in enumerate(questions, 1):
            print(f"\n[{processed + 1}/{total_questions}] {question[:80]}...")
            try:
                result = process_single_query(question, category=category, k=k)
                all_results.append(result)
                processed += 1
                print(
                    f"[OK] Completed (Total: {result['total_latency_ms']}ms, Chunks: {result['rag_chunks_count']})"
                )
            except Exception as e:
                logger.error(f"Failed to process question: {e}")
                # Add error result
                error_result = {
                    "query": question,
                    "category": category,
                    "retrieval_context": "",
                    "final_answer": f"Error: {str(e)}",
                    "retrieval_latency_ms": 0,
                    "generation_latency_ms": 0,
                    "total_latency_ms": 0,
                    "token_usage": "",
                    "rag_chunks_count": 0,
                    "rag_error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                all_results.append(error_result)
                processed += 1
                print(f"[ERROR] Error occurred")

    # Save results
    if output_path:
        output_file = Path(output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            project_root / "test" / "outputs" / f"simple_rag_test_cases_{timestamp}.csv"
        )

    save_results_to_csv(all_results, output_file)

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total questions processed: {processed}/{total_questions}")
    print(f"Results saved to: {output_file}")
    error_count = sum(1 for r in all_results if r.get("rag_error"))
    print(f"Success rate: {((processed - error_count)/processed*100):.1f}%")
    print(f"Errors: {error_count}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Simple RAG: Retrieve chunks and generate answer, or generate test cases for DeepEval"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The query to answer (for simple query mode).",
    )
    parser.add_argument(
        "--mode",
        choices=["query", "interactive", "batch"],
        default="query",
        help="Mode: query (simple answer), interactive (single test case), batch (process CSV)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of chunks to retrieve (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="../questions.csv",
        help="Path to questions.csv file (for batch mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (for test case generation modes)",
    )

    args = parser.parse_args()

    # Simple query mode (original behavior)
    if args.mode == "query":
        query = args.query or "what is attention?"

        logger.info("=" * 60)
        logger.info("Simple RAG System - Query Mode")
        logger.info("=" * 60)
        logger.info(f"Query: {query}")
        logger.info(f"Retrieval k: {args.k}")
        logger.info(f"LLM Model: {args.model}")

        total_start = time.perf_counter()

        # 1. Load vector store
        logger.info("\n[Step 1] Loading vector store...")
        vector_store_local = load_vector_store()
        if not vector_store_local:
            logger.error("Failed to load vector store. Exiting.")
            return

        # 2. Initialize LLM
        logger.info(f"\n[Step 2] Initializing LLM ({args.model})...")
        try:
            llm_local = ChatGoogleGenerativeAI(
                model=args.model,
                temperature=0.8,
                convert_system_message_to_human=True,
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            return

        # 3. Retrieve chunks
        logger.info(f"\n[Step 3] Retrieving top {args.k} chunks...")
        retrieval_start = time.perf_counter()
        docs = retrieve_chunks(vector_store_local, query, k=args.k)
        retrieval_time = time.perf_counter() - retrieval_start

        if not docs:
            logger.warning("No chunks retrieved. Cannot generate answer.")
            return

        logger.info(f"Retrieved {len(docs)} chunks in {retrieval_time:.2f}s")
        logger.info("\nRetrieved chunks summary:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            content_preview = (
                doc.page_content[:100] + "..."
                if len(doc.page_content) > 100
                else doc.page_content
            )
            logger.info(f"  [{i}] Source: {source}, Chunk ID: {chunk_id}")
            logger.info(f"      Preview: {content_preview}")

        # 4. Format context
        logger.info("\n[Step 4] Formatting context...")
        context = format_context(docs, max_chars_per_chunk=500)

        # 5. Generate answer
        logger.info("\n[Step 5] Generating answer...")
        generation_start = time.perf_counter()
        answer, _ = generate_answer(query, context, llm_local)
        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - total_start

        # 6. Display results
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Retrieval time: {retrieval_time:.2f}s")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print(f"Chunks retrieved: {len(docs)}")
        print("=" * 60)

    # Test case generation modes
    elif args.mode == "interactive":
        if args.query:
            # Process single query from command line
            result = process_single_query(args.query, k=args.k)
            output_path = (
                Path(args.output)
                if args.output
                else project_root
                / "test"
                / "outputs"
                / "simple_rag_test_case_single.csv"
            )
            save_results_to_csv([result], output_path)
            print(f"\nResults saved to {output_path}")
            print(f"\nFinal Answer:\n{result['final_answer']}")
        else:
            interactive_mode()

    elif args.mode == "batch":
        batch_mode(args.csv_path, args.output, k=args.k)


if __name__ == "__main__":
    main()
