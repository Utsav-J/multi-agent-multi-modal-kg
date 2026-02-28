from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / ".cursor" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "static_architecture_audit.md"

COUPLING_MODULES = {"agents", "utils", "knowledge_graph", "test", "chunking_strategy"}

FOCUS_PATTERNS = [
    "gemini-2.5-flash",
    "google/embeddinggemma-300m",
    "vector_store_outputs",
    "chunking_outputs",
    "knowledge_graph_outputs",
    "validation_outputs",
    "entityIndex",
]


def top_module_for(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT)
    except Exception:
        return "other"
    return rel.parts[0] if rel.parts else "other"


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


@dataclass
class FileScan:
    file: Path
    top_module: str
    imports: set[str]
    env_vars: list[str]
    argparse_args: list[str]


def extract_import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def extract_env_vars(tree: ast.AST) -> list[str]:
    keys: list[str] = []

    def is_os_getenv_call(n: ast.Call) -> bool:
        return (
            isinstance(n.func, ast.Attribute)
            and n.func.attr == "getenv"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "os"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_os_getenv_call(node):
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.append(node.args[0].value)
    return keys


def extract_argparse_args(tree: ast.AST) -> list[str]:
    args: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                args.append(node.args[0].value)
    return args


def main() -> int:
    py_files = [
        p
        for p in ROOT.rglob("*.py")
        if ".venv" not in p.parts and "node_modules" not in p.parts
    ]

    scans: list[FileScan] = []
    parse_failures: list[tuple[Path, str]] = []

    for f in py_files:
        txt = safe_read(f)
        try:
            tree = ast.parse(txt)
        except Exception as e:
            parse_failures.append((f, str(e)))
            continue
        scans.append(
            FileScan(
                file=f,
                top_module=top_module_for(f),
                imports=extract_import_roots(tree),
                env_vars=extract_env_vars(tree),
                argparse_args=extract_argparse_args(tree),
            )
        )

    # Coupling edges by import occurrences
    imports_from_module: dict[str, Counter[str]] = {m: Counter() for m in COUPLING_MODULES}
    for s in scans:
        src = s.top_module
        if src not in COUPLING_MODULES:
            continue
        for imp in s.imports:
            if imp in COUPLING_MODULES:
                imports_from_module[src][imp] += 1

    Ce = {m: len({k for k in imports_from_module[m].keys() if k != m}) for m in COUPLING_MODULES}
    Ca = {m: 0 for m in COUPLING_MODULES}
    for src in COUPLING_MODULES:
        for dst in imports_from_module[src].keys():
            if dst in COUPLING_MODULES and dst != src:
                Ca[dst] += 1
    Instability = {m: (Ce[m] / (Ce[m] + Ca[m]) if (Ce[m] + Ca[m]) else 0.0) for m in COUPLING_MODULES}

    edge_counts = Counter()
    for src in COUPLING_MODULES:
        for dst, c in imports_from_module[src].items():
            if dst != src:
                edge_counts[(src, dst)] += c

    env_counter = Counter()
    arg_counter = Counter()
    for s in scans:
        env_counter.update(s.env_vars)
        for a in s.argparse_args:
            arg_counter[(s.top_module, a)] += 1

    focused_hits: dict[str, list[str]] = {k: [] for k in FOCUS_PATTERNS}
    for s in scans:
        content = safe_read(s.file)
        for pat in FOCUS_PATTERNS:
            if pat in content:
                focused_hits[pat].append(str(s.file.relative_to(ROOT)))

    referenced_missing: list[tuple[str, str]] = []
    missing_targets = [
        ("main_pipeline.py", "Referenced in README.md/SETUP.md but missing"),
        ("agents/QUERY_AGENT_BEHAVIOR.md", "Referenced in README.md and agents/6-query_agent.md but missing"),
    ]
    for target, note in missing_targets:
        if not (ROOT / target).exists():
            referenced_missing.append((target, note))

    obs = {
        "query_agent_logs": "logs/query_agent_logs.txt",
        "deepeval_runner": "test/run_deepeval.py",
        "testcase_generator": "test/generate_test_cases.py",
        "deepeval_logs_dir": "test/logs",
        "test_cases_dir": "test/test_cases",
    }
    obs_present = {k: (ROOT / p).exists() for k, p in obs.items()}

    lines: list[str] = []
    lines += [
        "# Static Architecture Audit (No Pipeline Execution)",
        "",
        f"- **Repo**: `{ROOT.name}`",
        f"- **Generated**: `{REPORT_PATH.relative_to(ROOT)}`",
        f"- **Python files scanned**: **{len(scans)}** (parse failures: {len(parse_failures)})",
        "",
        "This report was generated by *static scanning only* (AST parse + text search). No agents were executed; no Neo4j/FAISS/LLM calls were made.",
        "",
        "---",
        "",
        "## 1) Architecture hygiene / repo mismatches (static)",
        "",
    ]

    if referenced_missing:
        lines += ["**Missing referenced artifacts**", ""]
        for tgt, note in referenced_missing:
            lines += [f"- **`{tgt}`**: {note}"]
    else:
        lines += ["- No missing referenced artifacts detected from the configured check list."]

    lines += [
        "",
        "---",
        "",
        "## 2) Module coupling (Ca/Ce/Instability) — static imports",
        "",
        "Top-level modules treated as components: `agents/`, `utils/`, `knowledge_graph/`, `test/`, `chunking_strategy/`.",
        "",
        "| module | Ca (imported-by modules) | Ce (imports other modules) | Instability I=Ce/(Ca+Ce) |",
        "|---|---:|---:|---:|",
    ]
    for m in sorted(COUPLING_MODULES):
        lines += [f"| `{m}` | {Ca[m]} | {Ce[m]} | {Instability[m]:.2f} |"]

    lines += ["", "**Hot coupling edges (by import occurrences)**", ""]
    for (src, dst), c in edge_counts.most_common(15):
        lines += [f"- **`{src} → {dst}`**: {c} import occurrences"]

    lines += [
        "",
        "---",
        "",
        "## 3) Config surface area (env vars + CLI args) — static",
        "",
        "### 3.1 Environment variables (from `os.getenv(\"...\")`)",
        "",
    ]
    if env_counter:
        for k, c in env_counter.most_common():
            lines += [f"- **`{k}`**: {c} occurrences"]
    else:
        lines += ["- No `os.getenv(\"...\")` keys detected."]

    lines += ["", "### 3.2 CLI args (from `argparse.add_argument(...)`)", ""]
    by_mod = defaultdict(list)
    for (mod, arg), c in sorted(arg_counter.items(), key=lambda x: (x[0][0], x[0][1])):
        by_mod[mod].append((arg, c))
    for mod in sorted(by_mod.keys()):
        lines += [f"- **`{mod}`**: " + ", ".join([f"`{a}` ({c})" for a, c in by_mod[mod]])]

    lines += [
        "",
        "---",
        "",
        "## 4) Hard-coded constants duplication (focused)",
        "",
        "These are repeated string constants that typically indicate opportunities to centralize configuration in `utils/` or a config module.",
        "",
    ]
    for pat in FOCUS_PATTERNS:
        files = focused_hits.get(pat, [])
        if not files:
            continue
        lines += [f"### `{pat}` ({len(files)} files)", ""]
        for f in sorted(files):
            lines += [f"- `{f}`"]
        lines += [""]

    lines += [
        "---",
        "",
        "## 5) Observability + testability inventory (static)",
        "",
        "| item | path | present? |",
        "|---|---|---|",
    ]
    for k, p in obs.items():
        lines += [f"| **{k}** | `{p}` | {'yes' if obs_present[k] else 'no'} |"]

    lines += [
        "",
        "---",
        "",
        "## 6) Follow-up questions (to finalize “basic-info-required” tests)",
        "",
        "Answering these lets us finalize the fault-injection matrix and acceptance criteria *without running the pipeline*:",
        "",
        "1) **Neo4j**: local Docker/Desktop/Aura? Is **APOC enabled**? Do you have the **full-text index `entityIndex`** created?",
        "2) **Corpus scale**: about how many PDFs, average pages per PDF, and typical image count per PDF?",
        "3) **Hardware target**: what machine specs do you consider the intended deployment (CPU/RAM)?",
    ]

    if parse_failures:
        lines += [
            "",
            "---",
            "",
            "## Appendix: parse failures",
            "",
            "These files could not be parsed by the AST scanner.",
            "",
        ]
        for f, err in parse_failures[:25]:
            lines += [f"- `{f.relative_to(ROOT)}`: `{err}`"]
        if len(parse_failures) > 25:
            lines += [f"- ... and {len(parse_failures) - 25} more"]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

