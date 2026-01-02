"""
Script to generate DeepEval test case CSV files from query agent results.

This script has two entry points:
1. Interactive mode: Test with a single question
2. Batch mode: Process all questions from questions.csv file
"""

import sys
import os
import csv
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root and test directory to path
project_root = Path(__file__).resolve().parent.parent
test_dir = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(test_dir))

# Import query agent components using importlib (file name starts with number)
import importlib.util

query_agent_path = project_root / "agents" / "6-query_agent.py"
spec = importlib.util.spec_from_file_location("query_agent", query_agent_path)
query_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_agent)

# Import functions and variables from query agent
rag_retrieval_tool = query_agent.rag_retrieval_tool
graph_retrieval_tool = query_agent.graph_retrieval_tool
load_vector_store = query_agent.load_vector_store
LAST_QUERY_TRACE = query_agent.LAST_QUERY_TRACE

from langchain_google_genai import ChatGoogleGenerativeAI

# Import CSV utilities
from csv_questions_utils import load_questions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global variables for query agent (use the module's globals)
def initialize_query_agent():
    """Initialize the query agent resources (LLM and vector store)."""
    # Use the query agent module's global variables
    if query_agent.llm_for_tools is None:
        logger.info("Initializing LLM for query agent...")
        query_agent.llm_for_tools = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        )

    if query_agent.vector_store is None:
        logger.info("Loading vector store...")
        query_agent.vector_store = load_vector_store()
        if not query_agent.vector_store:
            logger.warning("Vector store could not be loaded. RAG tool will fail.")


def process_single_query(query: str, category: Optional[str] = None) -> Dict[str, any]:
    """
    Process a single query through the query agent and return all relevant data.

    Args:
        query: The question to process
        category: Optional category name for the question

    Returns:
        Dictionary containing query, retrieval contexts, final answer, and metadata
    """
    logger.info(f"Processing query: {query}")

    # Initialize resources if needed
    initialize_query_agent()

    total_t0 = time.perf_counter()

    # 1) Call RAG tool
    logger.info("Invoking RAG retrieval tool...")
    rag_t0 = time.perf_counter()
    try:
        rag_result = rag_retrieval_tool.invoke(query)
        rag_ms = int((time.perf_counter() - rag_t0) * 1000)
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        rag_result = f"Error: {str(e)}"
        rag_ms = 0

    # 2) Call Graph tool
    logger.info("Invoking Graph retrieval tool...")
    graph_t0 = time.perf_counter()
    try:
        graph_result = graph_retrieval_tool.invoke(query)
        graph_ms = int((time.perf_counter() - graph_t0) * 1000)
    except Exception as e:
        logger.error(f"Graph retrieval failed: {e}")
        graph_result = f"Error: {str(e)}"
        graph_ms = 0

    # 3) Synthesize final answer
    synthesis_prompt = f"""
You are a helpful AI assistant that must answer user questions by combining two sources of information:

1. TEXT CONTEXT (RAG from vector store)
2. GRAPH CONTEXT (Neo4j knowledge graph)

User question:
{query}

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
    final_answer = ""
    synth_ms = 0
    token_usage = None

    try:
        synth_t0 = time.perf_counter()
        # Use query agent's LLM
        final_msg = query_agent.llm_for_tools.invoke(synthesis_prompt)
        synth_ms = int((time.perf_counter() - synth_t0) * 1000)
        final_answer = (
            final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        )

        # Extract token usage if available
        try:
            if hasattr(final_msg, "response_metadata"):
                meta = final_msg.response_metadata or {}
                token_usage = {
                    k: v
                    for k, v in meta.items()
                    if "token" in k.lower() and isinstance(v, (int, float))
                }
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        final_answer = f"Error during synthesis: {str(e)}"

    total_ms = int((time.perf_counter() - total_t0) * 1000)

    # Extract retrieval metadata from traces
    rag_trace = LAST_QUERY_TRACE.get("rag", {})
    graph_trace = LAST_QUERY_TRACE.get("graph", {})

    # Combine retrieval contexts
    combined_retrieval_context = f"""--- RAG CONTEXT ---
{rag_result}

--- GRAPH CONTEXT ---
{graph_result}"""

    # Build result dictionary
    result = {
        "query": query,
        "category": category or "",
        "rag_context": rag_result,
        "graph_context": graph_result,
        "retrieval_context": combined_retrieval_context,
        "final_answer": final_answer,
        "rag_latency_ms": rag_ms,
        "graph_latency_ms": graph_ms,
        "synthesis_latency_ms": synth_ms,
        "total_latency_ms": total_ms,
        "token_usage": str(token_usage) if token_usage else "",
        "rag_subqueries": str(rag_trace.get("rewritten_or_decomposed_queries", [])),
        "rag_chunks_count": len(rag_trace.get("retrieved_chunks", [])),
        "graph_entities_count": len(graph_trace.get("grounded_entities", [])),
        "graph_cypher": graph_trace.get("generated_cypher", ""),
        "rag_error": rag_trace.get("error", ""),
        "graph_error": graph_trace.get("error", ""),
        "timestamp": datetime.now().isoformat(),
    }

    return result


def save_results_to_csv(results: List[Dict[str, any]], output_path: Path):
    """
    Save results to a CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to save the CSV file
    """
    if not results:
        logger.warning("No results to save.")
        return

    # Define CSV columns
    fieldnames = [
        "query",
        "category",
        "rag_context",
        "graph_context",
        "retrieval_context",
        "final_answer",
        "rag_latency_ms",
        "graph_latency_ms",
        "synthesis_latency_ms",
        "total_latency_ms",
        "token_usage",
        "rag_subqueries",
        "rag_chunks_count",
        "graph_entities_count",
        "graph_cypher",
        "rag_error",
        "graph_error",
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
    print("INTERACTIVE MODE - Single Query Test")
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
    print(f"\nRAG Context (first 500 chars):\n{result['rag_context'][:500]}...")
    print(f"\nGraph Context (first 500 chars):\n{result['graph_context'][:500]}...")
    print(f"\nFinal Answer:\n{result['final_answer']}")
    print(
        f"\nLatencies: RAG={result['rag_latency_ms']}ms, "
        f"Graph={result['graph_latency_ms']}ms, "
        f"Synthesis={result['synthesis_latency_ms']}ms, "
        f"Total={result['total_latency_ms']}ms"
    )

    # Ask if user wants to save
    save = input("\nSave to CSV? (y/n): ").strip().lower()
    if save == "y":
        output_path = project_root / "test" / "outputs" / "test_case_single.csv"
        save_results_to_csv([result], output_path)
        print(f"Saved to {output_path}")


def batch_mode(csv_path: str, output_path: Optional[str] = None):
    """
    Batch mode: Process all questions from questions.csv file.

    Args:
        csv_path: Path to the questions.csv file
        output_path: Optional custom output path for the results CSV
    """
    print("\n" + "=" * 80)
    print("BATCH MODE - Processing Questions from CSV")
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
                result = process_single_query(question, category=category)
                all_results.append(result)
                processed += 1
                print(f"✓ Completed (Total: {result['total_latency_ms']}ms)")
            except Exception as e:
                logger.error(f"Failed to process question: {e}")
                # Add error result
                error_result = {
                    "query": question,
                    "category": category,
                    "rag_context": "",
                    "graph_context": "",
                    "retrieval_context": "",
                    "final_answer": f"Error: {str(e)}",
                    "rag_latency_ms": 0,
                    "graph_latency_ms": 0,
                    "synthesis_latency_ms": 0,
                    "total_latency_ms": 0,
                    "token_usage": "",
                    "rag_subqueries": "",
                    "rag_chunks_count": 0,
                    "graph_entities_count": 0,
                    "graph_cypher": "",
                    "rag_error": str(e),
                    "graph_error": "",
                    "timestamp": datetime.now().isoformat(),
                }
                all_results.append(error_result)
                processed += 1
                print(f"✗ Error occurred")

    # Save results
    if output_path:
        output_file = Path(output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = project_root / "test" / "outputs" / f"test_cases_{timestamp}.csv"

    save_results_to_csv(all_results, output_file)

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total questions processed: {processed}/{total_questions}")
    print(f"Results saved to: {output_file}")
    print(
        f"Success rate: {(processed - sum(1 for r in all_results if r.get('rag_error') or r.get('graph_error')))/processed*100:.1f}%"
    )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate DeepEval test case CSV from query agent results"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="Mode to run: interactive (single query) or batch (process CSV)",
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
        help="Output CSV file path (optional, auto-generated if not provided)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process (for interactive mode, overrides prompt)",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        if args.query:
            # Process single query from command line
            result = process_single_query(args.query)
            output_path = (
                Path(args.output)
                if args.output
                else project_root / "test" / "outputs" / "test_case_single.csv"
            )
            save_results_to_csv([result], output_path)
            print(f"\nResults saved to {output_path}")
            print(f"\nFinal Answer:\n{result['final_answer']}")
        else:
            interactive_mode()
    else:  # batch mode
        batch_mode(args.csv_path, args.output)


if __name__ == "__main__":
    main()
