"""
DeepEval evaluation for MAS vs GraphRAG comparison results.

Runs query+answer-only metrics (no retrieval context) on both mas_answer and
graphrag_answer from a comparison CSV produced by benchmarks/compare.py.
Outputs a CSV with per-metric scores for each system.

Usage
-----
    uv run benchmarks/evaluate_deepeval.py --input benchmarks/results/comparison_YYYYMMDD_HHMMSS.csv

    uv run benchmarks/evaluate_deepeval.py --input comparison.csv --out evaluation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

# Increase CSV field size limit for large answer columns
_csv_max = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_max)
        break
    except OverflowError:
        _csv_max = int(_csv_max / 10)

from deepeval.test_case import LLMTestCase
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.bias.bias import BiasMetric
from deepeval.metrics.toxicity.toxicity import ToxicityMetric
from deepeval.test_case.llm_test_case import LLMTestCaseParams
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.models.llms.gemini_model import GeminiModel


def _load_metric_criteria(metrics_file: Path, key: str, default: str = "") -> str:
    try:
        import json
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Query+answer-only metrics (no retrieval context)
# ---------------------------------------------------------------------------

def _create_metrics(model: GeminiModel, metrics_config: Path):
    """Create metrics that only require input (query) and actual_output (answer)."""
    tone_crit = _load_metric_criteria(metrics_config, "TONE", "Professional, respectful tone.")
    completeness_crit = _load_metric_criteria(metrics_config, "COMPLETENESS", "Fully addresses the query.")
    clarity_crit = _load_metric_criteria(metrics_config, "CLARITY", "Clear and easy to understand.")

    return [
        AnswerRelevancyMetric(model=model, threshold=0.5),
        BiasMetric(model=model),
        ToxicityMetric(model=model),
        GEval(
            name="Clarity",
            criteria=clarity_crit,
            model=model,
            threshold=0.6,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        ),
        GEval(
            name="Completeness",
            criteria=completeness_crit,
            model=model,
            threshold=0.5,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        ),
        GEval(
            name="Tone",
            criteria=tone_crit,
            model=model,
            threshold=0.6,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        ),
    ]


def _evaluate_single(
    metric, test_case: LLMTestCase
) -> tuple[float | None, bool | None, str]:
    """Run a single metric on a test case. Returns (score, pass, reason)."""
    try:
        result = metric.measure(test_case)
        score = getattr(result, "score", None) or getattr(metric, "score", None)
        success = getattr(result, "success", None)
        reason = getattr(result, "reason", "") or ""

        if success is None and score is not None:
            threshold = getattr(metric, "threshold", 0.5)
            success = score >= threshold

        return (
            float(score) if score is not None else None,
            bool(success) if success is not None else None,
            str(reason)[:500],
        )
    except Exception as e:
        return None, None, f"Error: {str(e)[:400]}"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _get_metric_name(metric) -> str:
    if hasattr(metric, "name"):
        return getattr(metric, "name", "")
    return metric.__class__.__name__.replace("Metric", "")


def _get_csv_columns(metrics: list) -> list[str]:
    base = ["query_id", "category", "query"]
    for m in metrics:
        name = _get_metric_name(m)
        base.extend([
            f"mas_{name}_score", f"mas_{name}_pass", f"mas_{name}_reason",
            f"graphrag_{name}_score", f"graphrag_{name}_pass", f"graphrag_{name}_reason",
        ])
    return base


def _write_header(path: Path, columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()


def _append_row(path: Path, row: dict, columns: list[str]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run DeepEval (query+answer-only) on MAS vs GraphRAG comparison CSV"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to comparison CSV from benchmarks/compare.py",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: benchmarks/results/evaluation_<timestamp>.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to evaluate (for testing)",
    )

    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or (
        RESULTS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_config = PROJECT_ROOT / "test" / "configs" / "metrics.json"
    model = GeminiModel(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )
    metrics = _create_metrics(model, metrics_config)
    columns = _get_csv_columns(metrics)

    _write_header(out_path, columns)
    print(f"Output: {out_path}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit:
        rows = rows[: args.limit]
        print(f"Limited to {len(rows)} rows.\n")

    total = len(rows)
    for idx, r in enumerate(rows, 1):
        qid = r.get("query_id", "")
        category = r.get("category", "")
        query = r.get("query", "")
        mas_answer = r.get("mas_answer", "")
        graphrag_answer = r.get("graphrag_answer", "")

        if not query or (not mas_answer and not graphrag_answer):
            print(f"[{idx}/{total}] Skipping (missing query or answers): {query[:60]}...")
            continue

        print(f"[{idx}/{total}] {query[:70]}...")

        row_out = {
            "query_id": qid,
            "category": category,
            "query": query[:2000],  # truncate for CSV readability
        }

        for metric in metrics:
            name = _get_metric_name(metric)

            if mas_answer:
                tc_mas = LLMTestCase(input=query, actual_output=mas_answer, retrieval_context=None)
                sc, pas, reason = _evaluate_single(metric, tc_mas)
                row_out[f"mas_{name}_score"] = sc if sc is not None else ""
                row_out[f"mas_{name}_pass"] = pas if pas is not None else ""
                row_out[f"mas_{name}_reason"] = reason or ""
            else:
                row_out[f"mas_{name}_score"] = row_out[f"mas_{name}_pass"] = row_out[f"mas_{name}_reason"] = ""

            if graphrag_answer:
                tc_gr = LLMTestCase(input=query, actual_output=graphrag_answer, retrieval_context=None)
                sc, pas, reason = _evaluate_single(metric, tc_gr)
                row_out[f"graphrag_{name}_score"] = sc if sc is not None else ""
                row_out[f"graphrag_{name}_pass"] = pas if pas is not None else ""
                row_out[f"graphrag_{name}_reason"] = reason or ""
            else:
                row_out[f"graphrag_{name}_score"] = row_out[f"graphrag_{name}_pass"] = row_out[f"graphrag_{name}_reason"] = ""

        _append_row(out_path, row_out, columns)

    print(f"\nDone. Results written to: {out_path}")


if __name__ == "__main__":
    main()
