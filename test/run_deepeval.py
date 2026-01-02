"""
DeepEval Test Runner for RAG/Graph Query Agent Evaluation

This script reads test case CSV files and runs comprehensive DeepEval evaluations
on all test cases synchronously, logging results to a separate file.
"""

import sys
import os
import csv
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
test_dir = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(test_dir))

# Import custom metrics
from reference_custom_metrics import metrics

# Configure logging
logs_dir = test_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = (
    logs_dir / f"deepeval_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# Set up logging to both file and console
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
console_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, console_handler],
)
logger = logging.getLogger(__name__)


def load_test_cases_from_csv(
    csv_path: Path, limit: Optional[int] = None
) -> List[LLMTestCase]:
    """
    Load test cases from CSV file and convert to LLMTestCase objects.

    Args:
        csv_path: Path to the CSV file containing test cases
        limit: Optional limit on number of test cases to load (for testing)

    Returns:
        List of LLMTestCase objects
    """
    test_cases = []
    logger.info(f"Loading test cases from {csv_path}")

    try:
        # Increase CSV field size limit to handle large text fields (retrieval contexts, answers)
        max_int = sys.maxsize
        while True:
            # Decrease the maxInt value by factor 10 as long as the OverflowError occurs
            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if limit:
                rows = rows[:limit]
                logger.info(f"Limited to first {limit} test cases")

            for idx, row in enumerate(rows, 1):
                try:
                    query = row.get("query", "").strip()
                    final_answer = row.get("final_answer", "").strip()
                    retrieval_context = row.get("retrieval_context", "").strip()
                    rag_context = row.get("rag_context", "").strip()
                    graph_context = row.get("graph_context", "").strip()
                    category = row.get("category", "").strip()

                    if not query or not final_answer:
                        logger.warning(
                            f"Row {idx}: Skipping due to missing query or final_answer"
                        )
                        continue

                    # Use retrieval_context if available, otherwise combine rag and graph contexts
                    if not retrieval_context:
                        if rag_context and graph_context:
                            retrieval_context = f"--- RAG CONTEXT ---\n{rag_context}\n\n--- GRAPH CONTEXT ---\n{graph_context}"
                        elif rag_context:
                            retrieval_context = rag_context
                        elif graph_context:
                            retrieval_context = graph_context

                    # Convert retrieval_context to list of strings (DeepEval requirement)
                    # Split by double newlines to create logical chunks, or use the whole string as one item
                    retrieval_context_list = None
                    if retrieval_context:
                        # Split by double newlines to create meaningful chunks
                        chunks = [
                            chunk.strip()
                            for chunk in retrieval_context.split("\n\n")
                            if chunk.strip()
                        ]
                        if chunks:
                            retrieval_context_list = chunks
                        else:
                            # If no double newlines, use the whole string as a single item
                            retrieval_context_list = [retrieval_context]

                    # Store metadata separately for later use
                    test_case_metadata = {
                        "category": category,
                        "rag_context": (
                            rag_context[:500] if rag_context else ""
                        ),  # Truncate for metadata
                        "graph_context": (graph_context[:500] if graph_context else ""),
                        "row_index": idx,
                        "rag_chunks_count": row.get("rag_chunks_count", ""),
                        "graph_entities_count": row.get("graph_entities_count", ""),
                        "total_latency_ms": row.get("total_latency_ms", ""),
                        "timestamp": row.get("timestamp", ""),
                    }

                    # Create LLMTestCase with all available context
                    test_case = LLMTestCase(
                        input=query,
                        actual_output=final_answer,
                        retrieval_context=retrieval_context_list,
                        expected_output=None,  # We don't have expected outputs
                    )

                    # Attach metadata as a custom attribute
                    test_case._custom_metadata = test_case_metadata
                    test_cases.append(test_case)

                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    continue

        logger.info(f"Successfully loaded {len(test_cases)} test cases")
        return test_cases

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}", exc_info=True)
        return []


def run_evaluation(
    test_cases: List[LLMTestCase],
    metrics: List,
    output_file: Optional[Path] = None,
) -> Dict:
    """
    Run DeepEval evaluation on test cases synchronously.

    Args:
        test_cases: List of LLMTestCase objects
        metrics: List of metric objects to evaluate
        output_file: Optional path to save detailed results JSON

    Returns:
        Dictionary containing evaluation results
    """
    logger.info("=" * 80)
    logger.info("Starting DeepEval Evaluation")
    logger.info("=" * 80)
    logger.info(f"Number of test cases: {len(test_cases)}")
    logger.info(f"Number of metrics: {len(metrics)}")
    logger.info(f"Metrics: {[m.__class__.__name__ for m in metrics]}")

    all_results = []
    summary_stats = {
        "total_test_cases": len(test_cases),
        "metrics_evaluated": len(metrics),
        "metric_names": [m.__class__.__name__ for m in metrics],
        "results_by_metric": {},
        "results_by_test_case": [],
    }

    # Initialize metric statistics
    for metric in metrics:
        metric_name = metric.__class__.__name__
        summary_stats["results_by_metric"][metric_name] = {
            "passed": 0,
            "failed": 0,
            "scores": [],
            "errors": 0,
        }

    # Process each test case
    for test_idx, test_case in enumerate(test_cases, 1):
        logger.info("\n" + "-" * 80)
        logger.info(
            f"Processing Test Case {test_idx}/{len(test_cases)}: {test_case.input[:80]}..."
        )

        # Safely get metadata (stored as custom attribute)
        metadata = getattr(test_case, "_custom_metadata", {})

        test_result = {
            "test_case_index": test_idx,
            "input": test_case.input,
            "category": metadata.get("category", ""),
            "metrics": {},
            "overall_passed": True,
        }

        # Evaluate with each metric synchronously
        for metric_idx, metric in enumerate(metrics, 1):
            metric_name = metric.__class__.__name__
            logger.info(f"  [{metric_idx}/{len(metrics)}] Evaluating: {metric_name}...")

            try:
                # Skip metrics that require context if context is missing
                # HallucinationMetric requires retrieval_context
                if metric_name == "HallucinationMetric":
                    has_context = (
                        test_case.retrieval_context is not None
                        and len(test_case.retrieval_context) > 0
                    )
                    if not has_context:
                        logger.warning(
                            f"    Skipping {metric_name}: retrieval_context is required but not available"
                        )
                        test_result["metrics"][metric_name] = {
                            "score": None,
                            "success": None,
                            "reason": "Skipped: retrieval_context required but not available",
                            "evaluation_time_seconds": 0,
                            "skipped": True,
                        }
                        summary_stats["results_by_metric"][metric_name]["errors"] += 1
                        continue

                # Measure evaluation time
                eval_start = datetime.now()

                # Run the metric evaluation
                metric_result = metric.measure(test_case)

                eval_time = (datetime.now() - eval_start).total_seconds()

                # Extract score and success status
                score = getattr(metric_result, "score", None)
                success = getattr(metric_result, "success", None)
                reason = getattr(metric_result, "reason", "")

                # If score is None, try to get it from the metric itself
                if score is None:
                    score = getattr(metric, "score", None)

                # If success is None, determine from score and threshold
                if success is None and score is not None:
                    threshold = getattr(metric, "threshold", 0.5)
                    success = score >= threshold

                test_result["metrics"][metric_name] = {
                    "score": float(score) if score is not None else None,
                    "success": bool(success) if success is not None else None,
                    "reason": str(reason) if reason else "",
                    "evaluation_time_seconds": eval_time,
                    "threshold": getattr(metric, "threshold", None),
                }

                # Update summary statistics
                if success:
                    summary_stats["results_by_metric"][metric_name]["passed"] += 1
                else:
                    summary_stats["results_by_metric"][metric_name]["failed"] += 1
                    test_result["overall_passed"] = False

                if score is not None:
                    summary_stats["results_by_metric"][metric_name]["scores"].append(
                        float(score)
                    )

                # Format score for logging
                score_str = f"{score:.3f}" if score is not None else "N/A"
                logger.info(
                    f"    Result: score={score_str}, "
                    f"success={success}, time={eval_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"    Error evaluating {metric_name}: {e}", exc_info=True)
                test_result["metrics"][metric_name] = {
                    "score": None,
                    "success": None,
                    "reason": f"Error: {str(e)}",
                    "evaluation_time_seconds": 0,
                    "error": True,
                }
                summary_stats["results_by_metric"][metric_name]["errors"] += 1
                test_result["overall_passed"] = False

        all_results.append(test_result)
        summary_stats["results_by_test_case"].append(test_result)

    # Calculate aggregate statistics
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)

    for metric_name, stats in summary_stats["results_by_metric"].items():
        total_evaluations = stats["passed"] + stats["failed"]
        pass_rate = (
            (stats["passed"] / total_evaluations * 100) if total_evaluations > 0 else 0
        )
        avg_score = (
            sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None
        )

        logger.info(f"\n{metric_name}:")
        logger.info(
            f"  Passed: {stats['passed']}/{total_evaluations} ({pass_rate:.1f}%)"
        )
        logger.info(f"  Failed: {stats['failed']}/{total_evaluations}")
        logger.info(f"  Errors: {stats['errors']}")
        if avg_score is not None:
            logger.info(f"  Average Score: {avg_score:.3f}")
        if stats["scores"]:
            logger.info(
                f"  Score Range: {min(stats['scores']):.3f} - {max(stats['scores']):.3f}"
            )

        # Update summary with calculated values
        stats["pass_rate"] = pass_rate
        stats["average_score"] = avg_score
        stats["min_score"] = min(stats["scores"]) if stats["scores"] else None
        stats["max_score"] = max(stats["scores"]) if stats["scores"] else None

    # Overall statistics
    total_passed_tests = sum(
        1 for tc in summary_stats["results_by_test_case"] if tc["overall_passed"]
    )
    overall_pass_rate = (
        (total_passed_tests / len(test_cases) * 100) if test_cases else 0
    )
    logger.info(f"\nOverall Results:")
    logger.info(f"  Total Test Cases: {len(test_cases)}")
    logger.info(f"  Fully Passed: {total_passed_tests} ({overall_pass_rate:.1f}%)")
    logger.info(f"  Partially Failed: {len(test_cases) - total_passed_tests}")

    summary_stats["overall_pass_rate"] = overall_pass_rate
    summary_stats["total_passed_tests"] = total_passed_tests

    # Save detailed results to JSON if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary_stats,
                    "detailed_results": all_results,
                    "evaluation_timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"\nDetailed results saved to: {output_file}")

    logger.info(f"\nEvaluation logs saved to: {log_file_path}")
    logger.info("=" * 80)

    return summary_stats


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run DeepEval evaluations on test case CSV files"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        default=str(test_dir / "outputs" / "test_case_single.csv"),
        help="Path to the CSV file containing test cases",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for detailed results (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate (for testing)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = test_dir / "outputs" / f"deepeval_results_{timestamp}.json"

    # Load test cases
    test_cases = load_test_cases_from_csv(csv_path, limit=args.limit)

    if not test_cases:
        logger.error("No test cases loaded. Exiting.")
        return

    # Run evaluation
    try:
        results = run_evaluation(test_cases, metrics, output_file)
        logger.info("\n✅ Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
