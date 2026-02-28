"""
Interpret, evaluate, and draw conclusions from DeepEval comparison results.

Reads an evaluation CSV (from benchmarks/evaluate_deepeval.py), computes
aggregate scores for MAS and GraphRAG, performs statistical analysis, and
produces a report with conclusions.

Usage
-----
    uv run benchmarks/interpret_results.py benchmarks/results/eval_test2.csv

    uv run benchmarks/interpret_results.py --input eval_test2.csv --out report.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

# Increase CSV field size limit
_csv_max = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_max)
        break
    except OverflowError:
        _csv_max = int(_csv_max / 10)

# Metrics where 0 = good (Bias, Toxicity). We invert so higher = better for aggregation.
INVERTED_METRICS = {"Bias", "Toxicity"}

# Weights for overall score (content quality metrics; Bias/Toxicity used inverted).
DEFAULT_WEIGHTS = {
    "AnswerRelevancy": 0.30,
    "Clarity": 0.20,
    "Completeness": 0.25,
    "Tone": 0.15,
    "Bias": 0.05,   # inverted: 0 bias -> 1.0
    "Toxicity": 0.05,  # inverted: 0 toxicity -> 1.0
}


def _parse_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_bool(val) -> bool | None:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def load_evaluation_csv(path: Path) -> tuple[list[dict], list[str], list[str]]:
    """Load evaluation CSV, return (rows, mas_score_cols, graphrag_score_cols)."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []

    mas_cols = [c for c in cols if c.startswith("mas_") and c.endswith("_score")]
    graphrag_cols = [c for c in cols if c.startswith("graphrag_") and c.endswith("_score")]

    # Align by metric name
    mas_by_metric = {c.replace("mas_", "").replace("_score", ""): c for c in mas_cols}
    graphrag_by_metric = {c.replace("graphrag_", "").replace("_score", ""): c for c in graphrag_cols}
    metrics = sorted(set(mas_by_metric) | set(graphrag_by_metric))

    mas_ordered = [mas_by_metric[m] for m in metrics if m in mas_by_metric]
    graphrag_ordered = [graphrag_by_metric[m] for m in metrics if m in graphrag_by_metric]

    return rows, mas_ordered, graphrag_ordered


def extract_scores(
    rows: list[dict],
    mas_cols: list[str],
    graphrag_cols: list[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[str]]:
    """
    Extract score arrays per metric for MAS and GraphRAG.
    Returns (mas_scores, graphrag_scores, metric_names).
    """
    metric_names = [c.replace("mas_", "").replace("_score", "") for c in mas_cols]
    mas_scores = {m: [] for m in metric_names}
    graphrag_scores = {m: [] for m in metric_names}

    for r in rows:
        for mas_c, gr_c in zip(mas_cols, graphrag_cols):
            m = mas_c.replace("mas_", "").replace("_score", "")
            v_mas = _parse_float(r.get(mas_c))
            v_gr = _parse_float(r.get(gr_c))
            if v_mas is not None:
                mas_scores[m].append(v_mas)
            if v_gr is not None:
                graphrag_scores[m].append(v_gr)

    return mas_scores, graphrag_scores, metric_names


def normalize_for_aggregation(scores: list[float], metric: str) -> list[float]:
    """For Bias/Toxicity, invert so 0 -> 1 (higher = better)."""
    if metric in INVERTED_METRICS:
        return [1.0 - s if s is not None else 0.0 for s in scores]
    return list(scores)


def compute_stats(values: list[float]) -> dict:
    """Compute mean, std, median, min, max, count."""
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "count": 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 0 else 0
    std = variance ** 0.5
    sorted_v = sorted(values)
    mid = n // 2
    median = (sorted_v[mid] + sorted_v[mid - 1]) / 2 if n % 2 == 0 else sorted_v[mid]
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": min(values),
        "max": max(values),
        "count": n,
    }


def weighted_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted mean of per-metric scores."""
    total_w = 0.0
    weighted_sum = 0.0
    for m, s in scores.items():
        w = weights.get(m, 0.0)
        if w > 0 and s is not None:
            weighted_sum += w * s
            total_w += w
    return weighted_sum / total_w if total_w > 0 else 0.0


def pass_rates(rows: list[dict], mas_cols: list[str], graphrag_cols: list[str]) -> dict:
    """Compute pass rate per metric for MAS and GraphRAG."""
    metric_names = [c.replace("mas_", "").replace("_score", "") for c in mas_cols]
    mas_pass_cols = [c.replace("_score", "_pass") for c in mas_cols]
    graphrag_pass_cols = [c.replace("_score", "_pass") for c in graphrag_cols]

    rates = {}
    for m, mp, gp in zip(metric_names, mas_pass_cols, graphrag_pass_cols):
        mas_pass = sum(1 for r in rows if _parse_bool(r.get(mp)) is True)
        gr_pass = sum(1 for r in rows if _parse_bool(r.get(gp)) is True)
        n = len(rows)
        rates[m] = {
            "mas_pass": mas_pass,
            "mas_rate": mas_pass / n if n else 0,
            "graphrag_pass": gr_pass,
            "graphrag_rate": gr_pass / n if n else 0,
        }
    return rates


def paired_t_test(mas: list[float], graphrag: list[float]) -> tuple[float, float]:
    """
    Paired t-test for H0: mean(mas - graphrag) = 0.
    Returns (t_statistic, p_value_approx).
    """
    n = len(mas)
    if n != len(graphrag) or n < 2:
        return float("nan"), float("nan")

    diffs = [a - b for a, b in zip(mas, graphrag)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0
    std_diff = var_diff ** 0.5
    if std_diff == 0:
        return 0.0, 1.0
    t = mean_diff / (std_diff / (n ** 0.5))

    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t), n - 1))
    except ImportError:
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p


def cohens_d(mas: list[float], graphrag: list[float]) -> float:
    """Cohen's d for paired samples (uses std of differences)."""
    n = len(mas)
    if n != len(graphrag) or n < 2:
        return float("nan")
    diffs = [a - b for a, b in zip(mas, graphrag)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std = var ** 0.5
    return mean_diff / std if std > 0 else 0.0


def build_report(
    rows: list[dict],
    mas_scores: dict[str, list[float]],
    graphrag_scores: dict[str, list[float]],
    metric_names: list[str],
    weights: dict[str, float],
) -> str:
    """Build a markdown report."""
    n = len(rows)
    categories = list({r.get("category", "") for r in rows if r.get("category")})

    lines = [
        "# DeepEval Comparison Report: MAS vs GraphRAG",
        "",
        f"**Total queries:** {n}",
        f"**Categories:** {', '.join(categories)}",
        "",
        "---",
        "",
        "## 1. Per-Metric Summary",
        "",
        "| Metric | MAS Mean | MAS Std | GR Mean | GR Std | MAS Median | GR Median | Diff (MAS-GR) |",
        "|--------|----------|---------|---------|--------|------------|-----------|------------|",
    ]

    mas_overall_scores = {}
    gr_overall_scores = {}

    for m in metric_names:
        mas_raw = mas_scores.get(m, [])
        gr_raw = graphrag_scores.get(m, [])
        mas_norm = normalize_for_aggregation(mas_raw, m)
        gr_norm = normalize_for_aggregation(gr_raw, m)

        mas_st = compute_stats(mas_norm)
        gr_st = compute_stats(gr_norm)
        delta = (mas_st["mean"] or 0) - (gr_st["mean"] or 0)

        mas_overall_scores[m] = mas_st["mean"]
        gr_overall_scores[m] = gr_st["mean"]

        lines.append(
            f"| {m} | {mas_st['mean']:.3f} | {mas_st['std']:.3f} | "
            f"{gr_st['mean']:.3f} | {gr_st['std']:.3f} | "
            f"{mas_st['median']:.3f} | {gr_st['median']:.3f} | "
            f"{delta:+.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Overall Scores (Weighted Mean)",
        "",
        f"Weights: {weights}",
        "",
    ])

    mas_overall = weighted_mean(mas_overall_scores, weights)
    gr_overall = weighted_mean(gr_overall_scores, weights)
    delta_overall = mas_overall - gr_overall

    lines.extend([
        f"- **MAS overall:** {mas_overall:.3f}",
        f"- **GraphRAG overall:** {gr_overall:.3f}",
        f"- **Difference (MAS - GraphRAG):** {delta_overall:+.3f}",
        "",
        "---",
        "",
        "## 3. Statistical Comparison",
        "",
    ])

    # Per-metric paired comparison
    lines.append("### Per-metric paired difference (MAS - GraphRAG)")
    lines.append("")
    lines.append("| Metric | Mean Diff | t | Cohen's d |")
    lines.append("|--------|--------|---|-----------|")

    for m in metric_names:
        mas_raw = mas_scores.get(m, [])
        gr_raw = graphrag_scores.get(m, [])
        mas_norm = normalize_for_aggregation(mas_raw, m)
        gr_norm = normalize_for_aggregation(gr_raw, m)
        if len(mas_norm) == len(gr_norm) and len(mas_norm) >= 2:
            t, p = paired_t_test(mas_norm, gr_norm)
            d = cohens_d(mas_norm, gr_norm)
            mean_d = (sum(mas_norm) - sum(gr_norm)) / len(mas_norm)
            lines.append(f"| {m} | {mean_d:+.3f} | {t:.2f} (p~{p:.3f}) | {d:.2f} |")
        else:
            lines.append(f"| {m} | — | — | — |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Pass Rates",
        "",
    ])

    mas_cols = [f"mas_{m}_score" for m in metric_names]
    graphrag_cols = [f"graphrag_{m}_score" for m in metric_names]
    rates = pass_rates(rows, mas_cols, graphrag_cols)

    lines.append("| Metric | MAS Pass | MAS Rate | GR Pass | GR Rate |")
    lines.append("|--------|----------|----------|---------|---------|")
    for m in metric_names:
        r = rates.get(m, {})
        lines.append(
            f"| {m} | {r.get('mas_pass', 0)}/{n} | {r.get('mas_rate', 0):.1%} | "
            f"{r.get('graphrag_pass', 0)}/{n} | {r.get('graphrag_rate', 0):.1%} |"
        )

    lines.extend([
        "",
        "_Note: For Bias and Toxicity, raw score 0 means no issues (ideal). "
        "Overall scores use inverted values (1 - raw) so higher = better._",
        "",
        "---",
        "",
        "## 5. Category-wise Breakdown & Comparison",
        "",
    ])

    # Group rows by category
    by_category = defaultdict(list)
    for r in rows:
        cat = r.get("category", "").strip() or "(uncategorized)"
        by_category[cat].append(r)

    for cat in sorted(by_category.keys()):
        cat_rows = by_category[cat]
        n_cat = len(cat_rows)
        lines.append(f"### {cat} (n={n_cat})")
        lines.append("")

        # Per-metric means for this category
        cat_mas, cat_gr, _ = extract_scores(cat_rows, mas_cols, graphrag_cols)
        cat_mas_means = {}
        cat_gr_means = {}
        for m in metric_names:
            mas_raw = cat_mas.get(m, [])
            gr_raw = cat_gr.get(m, [])
            mas_n = normalize_for_aggregation(mas_raw, m)
            gr_n = normalize_for_aggregation(gr_raw, m)
            cat_mas_means[m] = sum(mas_n) / len(mas_n) if mas_n else None
            cat_gr_means[m] = sum(gr_n) / len(gr_n) if gr_n else None

        # Per-metric table
        lines.append("| Metric | MAS Mean | GraphRAG Mean | Diff (MAS−GR) | Winner |")
        lines.append("|--------|----------|---------------|---------------|--------|")
        mas_wins_cat = gr_wins_cat = 0
        for m in metric_names:
            ma = cat_mas_means.get(m)
            gr = cat_gr_means.get(m)
            if ma is None and gr is None:
                lines.append(f"| {m} | — | — | — | — |")
                continue
            ma = ma if ma is not None else 0.0
            gr = gr if gr is not None else 0.0
            diff = ma - gr
            if diff > 0:
                winner = "MAS"
                mas_wins_cat += 1
            elif diff < 0:
                winner = "GraphRAG"
                gr_wins_cat += 1
            else:
                winner = "Tie"
            lines.append(f"| {m} | {ma:.3f} | {gr:.3f} | {diff:+.3f} | {winner} |")

        mas_overall_cat = weighted_mean(cat_mas_means, weights)
        gr_overall_cat = weighted_mean(cat_gr_means, weights)
        diff_overall = mas_overall_cat - gr_overall_cat
        winner_cat = "MAS" if diff_overall > 0 else ("GraphRAG" if diff_overall < 0 else "Tie")
        lines.extend([
            "",
            f"**Overall (weighted):** MAS = {mas_overall_cat:.3f}, GraphRAG = {gr_overall_cat:.3f}, "
            f"Diff = {diff_overall:+.3f} → **{winner_cat}**",
            "",
            f"**Metric wins:** MAS {mas_wins_cat}, GraphRAG {gr_wins_cat}",
            "",
            "",
        ])

    # Summary comparison table across categories
    lines.append("### Category-level summary")
    lines.append("")
    lines.append("| Category | n | MAS Overall | GraphRAG Overall | Diff | Winner |")
    lines.append("|----------|---|-------------|------------------|------|--------|")
    for cat in sorted(by_category.keys()):
        cat_rows = by_category[cat]
        cat_mas, cat_gr, _ = extract_scores(cat_rows, mas_cols, graphrag_cols)
        cat_mas_means = {}
        cat_gr_means = {}
        for m in metric_names:
            mas_n = normalize_for_aggregation(cat_mas.get(m, []), m)
            gr_n = normalize_for_aggregation(cat_gr.get(m, []), m)
            cat_mas_means[m] = sum(mas_n) / len(mas_n) if mas_n else 0.0
            cat_gr_means[m] = sum(gr_n) / len(gr_n) if gr_n else 0.0
        mas_overall_cat = weighted_mean(cat_mas_means, weights)
        gr_overall_cat = weighted_mean(cat_gr_means, weights)
        diff_overall = mas_overall_cat - gr_overall_cat
        winner_cat = "MAS" if diff_overall > 0 else ("GraphRAG" if diff_overall < 0 else "Tie")
        lines.append(f"| {cat} | {len(cat_rows)} | {mas_overall_cat:.3f} | {gr_overall_cat:.3f} | {diff_overall:+.3f} | {winner_cat} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Conclusions",
        "",
    ])

    winner = "MAS" if delta_overall > 0 else ("GraphRAG" if delta_overall < 0 else "Tie")
    lines.append(f"1. **Overall winner:** {winner} (overall score diff = {delta_overall:+.3f})")
    lines.append("")

    mas_wins = sum(1 for m in metric_names if (mas_overall_scores.get(m) or 0) > (gr_overall_scores.get(m) or 0))
    gr_wins = sum(1 for m in metric_names if (gr_overall_scores.get(m) or 0) > (mas_overall_scores.get(m) or 0))
    lines.append(f"2. **Metric-level:** MAS wins on {mas_wins} metrics, GraphRAG on {gr_wins}.")
    lines.append("")

    strong_diff = [m for m in metric_names if abs((mas_overall_scores.get(m) or 0) - (gr_overall_scores.get(m) or 0)) > 0.1]
    if strong_diff:
        lines.append(f"3. **Largest gaps:** {', '.join(strong_diff)}.")
    lines.append("")

    try:
        from scipy.stats import ttest_rel
        all_mas = []
        all_gr = []
        for m in metric_names:
            mas_n = normalize_for_aggregation(mas_scores.get(m, []), m)
            gr_n = normalize_for_aggregation(graphrag_scores.get(m, []), m)
            for a, b in zip(mas_n, gr_n):
                all_mas.append(a)
                all_gr.append(b)
        if len(all_mas) >= 2:
            t, p = ttest_rel(all_mas, all_gr)
            sig = "statistically significant" if p < 0.05 else "not statistically significant"
            lines.append(f"4. **Paired t-test (all scores):** {sig} (p = {p:.4f}).")
    except ImportError:
        lines.append("4. **Statistical test:** Install scipy for exact p-values.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Interpret DeepEval comparison results and produce overall scores"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=RESULTS_DIR / "eval_test2.csv",
        help="Path to evaluation CSV",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output report path (default: prints to stdout)",
    )

    args = parser.parse_args()
    input_path = args.input if isinstance(args.input, Path) else Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows, mas_cols, graphrag_cols = load_evaluation_csv(input_path)
    if not rows:
        print("Error: No rows in CSV.", file=sys.stderr)
        sys.exit(1)

    mas_scores, graphrag_scores, metric_names = extract_scores(rows, mas_cols, graphrag_cols)
    report = build_report(rows, mas_scores, graphrag_scores, metric_names, DEFAULT_WEIGHTS)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
