"""
Comparison harness: MAS Query Agent vs GraphRAG Local Search.

Runs a set of queries through both systems, collects answers + timing,
and writes the results to a CSV file under benchmarks/results/.

Usage
-----
Single query:
    uv run benchmarks/compare.py "What is the role of attention heads?"

Multiple queries from a file (one per line):
    uv run benchmarks/compare.py --file benchmarks/queries.txt

Options:
    --out <path>      Override the default CSV output path
    --graphrag-root   Path to graphrag workspace (default: graphrag_workspace)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GRAPHRAG_ROOT = PROJECT_ROOT / "graphrag_workspace"


def _utf8_env() -> dict[str, str]:
    """Return a copy of os.environ that forces UTF-8 on Windows."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------------------------------------------------------------------------
# MAS Query Agent
# ---------------------------------------------------------------------------


def run_mas_query(query: str) -> dict:
    """Run the MAS hybrid query agent via subprocess and parse output."""
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "agents" / "6-query_agent.py"),
        query,
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    wall_ms = int((time.perf_counter() - t0) * 1000)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    answer = ""
    trace_json: dict | None = None

    if "=== Final Answer ===" in stdout:
        after_answer = stdout.split("=== Final Answer ===", 1)[1]

        # Keep everything up to QUERY_TRACE_JSON (the log line that marks
        # the start of internal trace data we don't need in the answer).
        if "QUERY_TRACE_JSON" in after_answer:
            answer = after_answer.split("QUERY_TRACE_JSON", 1)[0].strip()
        elif "=== Query Trace (JSON) ===" in after_answer:
            answer = after_answer.split("=== Query Trace (JSON) ===", 1)[0].strip()
        else:
            answer = after_answer.strip()
    elif proc.returncode != 0:
        answer = f"[ERROR exit={proc.returncode}] {stderr[:500]}"

    internal_latency = None
    if trace_json and "latency_ms" in trace_json:
        internal_latency = trace_json["latency_ms"]

    return {
        "answer": answer,
        "wall_ms": wall_ms,
        "internal_latency": internal_latency,
        "trace": trace_json,
        "exit_code": proc.returncode,
        "error": stderr[:1000] if proc.returncode != 0 else "",
    }


# ---------------------------------------------------------------------------
# GraphRAG Local Search
# ---------------------------------------------------------------------------


def run_graphrag_query(query: str, root: Path = DEFAULT_GRAPHRAG_ROOT) -> dict:
    """Run GraphRAG local search via CLI subprocess and parse output."""
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "graphrag",
        "query",
        "--root",
        str(root),
        "--method",
        "local",
        query,
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    wall_ms = int((time.perf_counter() - t0) * 1000)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    answer = stdout.strip()

    if proc.returncode != 0 and not answer:
        answer = f"[ERROR exit={proc.returncode}] {stderr[:500]}"

    return {
        "answer": answer,
        "wall_ms": wall_ms,
        "exit_code": proc.returncode,
        "error": stderr[:1000] if proc.returncode != 0 else "",
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "query_id",
    "category",
    "query",
    "mas_answer",
    "graphrag_answer",
    "mas_wall_ms",
    "graphrag_wall_ms",
    "mas_rag_ms",
    "mas_graph_ms",
    "mas_synthesis_ms",
    "mas_exit_code",
    "graphrag_exit_code",
    "timestamp_utc",
]


def write_csv_header(path: Path) -> None:
    """Create the output file and write the CSV header."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
    print(f"\nOutput file: {path}")


def append_row_to_csv(path: Path, row: dict) -> None:
    """Append a single row to the CSV file."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare MAS Query Agent vs GraphRAG Local Search"
    )
    parser.add_argument(
        "queries",
        nargs="*",
        help="One or more queries to run (positional).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Path to a text file with one query per line.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Override CSV output path.",
    )
    parser.add_argument(
        "--graphrag-root",
        type=Path,
        default=DEFAULT_GRAPHRAG_ROOT,
        help="Root directory of the graphrag workspace.",
    )

    args = parser.parse_args()

    queries: list[str] = []
    categories: list[str] = []
    if args.file:
        file_path = args.file
        if file_path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            col = "Question" if "Question" in df.columns else df.columns[-1]
            queries = df[col].dropna().astype(str).tolist()
            if "Category" in df.columns:
                categories = df["Category"].fillna("").astype(str).tolist()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]
    if args.queries:
        queries.extend(args.queries)

    if not queries:
        parser.error("No queries provided. Pass queries as arguments or use --file.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (RESULTS_DIR / f"comparison_{timestamp}.csv")

    write_csv_header(out_path)
    rows: list[dict] = []
    total = len(queries)

    for idx, query in enumerate(queries, 1):
        qid = str(uuid.uuid4())[:8]
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] Query: {query}")
        print(f"{'='*70}")

        # --- MAS ---
        print("\n  Running MAS Query Agent...")
        mas = run_mas_query(query)
        print(f"  MAS finished in {mas['wall_ms']}ms (exit={mas['exit_code']})")
        if mas["exit_code"] != 0:
            print(f"  MAS error: {mas['error'][:200]}")

        # --- GraphRAG ---
        print("\n  Running GraphRAG Local Search...")
        gr = run_graphrag_query(query, root=args.graphrag_root)
        print(f"  GraphRAG finished in {gr['wall_ms']}ms (exit={gr['exit_code']})")
        if gr["exit_code"] != 0:
            print(f"  GraphRAG error: {gr['error'][:200]}")

        # --- Latency breakdown from MAS trace ---
        mas_rag_ms = ""
        mas_graph_ms = ""
        mas_synth_ms = ""
        if mas.get("internal_latency"):
            lat = mas["internal_latency"]
            mas_rag_ms = lat.get("rag", "")
            mas_graph_ms = lat.get("graph", "")
            mas_synth_ms = lat.get("synthesis", "")

        category = categories[idx - 1] if idx - 1 < len(categories) else ""

        row = {
            "query_id": qid,
            "category": category,
            "query": query,
            "mas_answer": mas["answer"],
            "graphrag_answer": gr["answer"],
            "mas_wall_ms": mas["wall_ms"],
            "graphrag_wall_ms": gr["wall_ms"],
            "mas_rag_ms": mas_rag_ms,
            "mas_graph_ms": mas_graph_ms,
            "mas_synthesis_ms": mas_synth_ms,
            "mas_exit_code": mas["exit_code"],
            "graphrag_exit_code": gr["exit_code"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        append_row_to_csv(out_path, row)

        # Print a short preview of both answers
        print(f"\n  --- MAS Answer (first 300 chars) ---")
        print(f"  {mas['answer'][:300]}")
        print(f"\n  --- GraphRAG Answer (first 300 chars) ---")
        print(f"  {gr['answer'][:300]}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Query':<50} {'MAS (ms)':>10} {'GR (ms)':>10}")
    print(f"{'-'*50} {'-'*10} {'-'*10}")
    for r in rows:
        q_short = r["query"][:47] + "..." if len(r["query"]) > 50 else r["query"]
        print(f"{q_short:<50} {r['mas_wall_ms']:>10} {r['graphrag_wall_ms']:>10}")


if __name__ == "__main__":
    main()
