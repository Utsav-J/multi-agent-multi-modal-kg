# Architecture Testing Plan (Translated to This Repo)

This document translates `.cursor/plans/architecture_testing.md` into **concrete, measurable architecture tests** for the current project: `MAS-for-multimodal-knowledge-graph`.

It is intentionally **system/architecture-level** (robustness, modularity, scalability, observability), not “model quality”. Where we need a “performance” proxy to compute ratios, we reuse the repo’s existing harnesses (`test/generate_test_cases.py`, `test/run_deepeval.py`) and existing logs/traces (notably Agent 6).

---

## 0) What “the architecture” is in *this* repo

### 0.1 Build-time pipeline (offline)

- **Agent 1** `agents/1-pdf_processor_agent.py`: PDF → Markdown (+ optional image captions)
- **Agent 2** `agents/2-chunker_agent.py`: Markdown → `*_chunks_5k.jsonl` + `*_chunks_2k.jsonl`
- **Agent 3** `agents/3-graph_data_extractor_agent.py`: `*_5k.jsonl` → `knowledge_graph_outputs/*_graph.jsonl` (plus deterministic image-subgraph extraction); maintains **`knowledge_graph_outputs/global_entity_registry.json`**
- **Agent 4** `agents/4-vector_store_creation_agent.py`: scans `chunking_outputs/*_2k.jsonl` → FAISS index at `vector_store_outputs/index`
- **Agent 5** `agents/5-jsonl_graph_ingestion_agent.py`: ingests `knowledge_graph_outputs/*.jsonl` → Neo4j (with APOC “upsert” best-effort)

### 0.2 Run-time query pipeline (online)

- **Agent 6** `agents/6-query_agent.py`: **hybrid retrieval** (FAISS RAG + Neo4j KG) → answer synthesis  
  - Already logs step latencies and keeps a **structured trace** in `LAST_QUERY_TRACE` (used by `test/generate_test_cases.py`).

### 0.3 Run-time validation pipeline (online / post-hoc)

- **Agent 7** `agents/7-validator_agent.py`: KG validation with:
  - deterministic structural checks (Neo4j Cypher queries)
  - LLM-based checks (coverage, faithfulness, plausibility)

### 0.4 Important repo reality checks (affects testing)

- **`main_pipeline.py` is referenced** in `README.md` and `SETUP.md`, but is **not present** in the workspace right now.  
  For architecture testing, treat the “orchestrator” as **the directed dependency graph of Agents 1→2→3→(4 & 5)**, executed manually or by an external runner.
- `README.md` references `agents/QUERY_AGENT_BEHAVIOR.md`, but that file is **missing**.  
  For reproducible architecture evaluation in a paper, the authoritative behavior spec should live somewhere (right now `agents/6-query_agent.md` is closest).

---

## 1) Architectural Robustness Metrics (translated)

### 1.1 Failure Isolation & Blast Radius

**Component graph (dependency edges)**:

- Build pipeline edges:
  - A1 → A2 → A3 → A5 → Neo4j
  - A2 → A4 → FAISS
  - A3 ↔ `global_entity_registry.json`
- Query edges:
  - A6 → (FAISS) and (Neo4j) → A6 synthesis
- Validation edges:
  - A7 → Neo4j (+ optional LLM)

**Metric implementations**

- **Failure Propagation Depth**: number of downstream components whose *primary function* becomes unavailable after a single injected failure.
- **Blast Radius Ratio**: \(\frac{\#\text{components affected}}{\#\text{components in the path}}\)
  - For a query run, the “path components” are: `{A6-RAG, A6-Graph, A6-Synthesis}`.
  - For build pipeline, the “path components” are: `{A1, A2, A3, A4, A5}` (plus stores).

**Fault-injection matrix (use as your experiment table)**

- **Neo4j unavailable**
  - Injection: wrong `NEO4J_URI` / stop Neo4j
  - Expected: A6 graph tool returns error string; A6 continues; A7 structural evaluator fails or reports errors
  - Measure: blast radius on `{A6-RAG, A6-Graph, A6-Synthesis}` and on `{A5}`
- **FAISS index missing**
  - Injection: rename `vector_store_outputs/index` temporarily
  - Expected: A6 RAG tool returns `"Error: Vector store is not available."`; A6 continues with graph-only
  - Measure: blast radius on query path; also “graceful degradation” (below)
- **Full-text entity index missing in Neo4j**
  - Injection: drop `entityIndex`
  - Expected: entity grounding returns no entities → graph retrieval returns “No relevant entities…”
  - Measure: graph path availability vs fallback
- **APOC missing**
  - Injection: run Neo4j without APOC plugin
  - Expected: Agent 5 falls back (it catches errors and continues ingestion with reduced upsert capability)
  - Measure: which sub-capability fails (metadata upsert) vs core ingestion succeeds
- **Gemini key missing**
  - Injection: unset `GOOGLE_API_KEY`
  - Expected: A1 annotation fails; A3 LLM extraction fails (text KG); A7 LLM-based checks fail; deterministic parts may still run
  - Measure: how much of the system remains usable (image subgraph extraction in A3 is deterministic)

**How to run and log**

- Query-path faults: run `test/generate_test_cases.py` in batch mode (so you get `rag_error`, `graph_error`, per-step latency, counts).
- Build-path faults: run agents individually and record which expected output artifacts still exist:
  - `markdown_outputs/*.md`
  - `chunking_outputs/*.jsonl`
  - `knowledge_graph_outputs/*_graph.jsonl` and registry
  - `vector_store_outputs/index`

### 1.2 Graceful Degradation Score (GDS)

In this repo, “performance under degraded mode” should be measured using a **consistent proxy**:

- **Proxy A (recommended)**: DeepEval scores from `test/run_deepeval.py` over a fixed test set.
- **Proxy B (cheap, deterministic)**: retrieval availability and evidence coverage:
  - `rag_chunks_count`
  - `graph_entities_count`
  - presence/absence of `rag_error` / `graph_error`

**GDS definition (use Proxy A for paper plots)**:

\[
GDS = \frac{\text{Mean DeepEval score in degraded mode}}{\text{Mean DeepEval score in full mode}}
\]

**Degraded modes that map cleanly to this system**

- **Graph-off**: make Neo4j unreachable → “RAG-only”
- **RAG-off**: remove FAISS index → “Graph-only”
- **LLM-off (partial)**: remove `GOOGLE_API_KEY` → evaluate how much deterministic sub-systems still work (A3 image subgraph; A7 structural checks)

**How to measure**

1) Generate a baseline CSV with `test/generate_test_cases.py` in “full mode”.  
2) Repeat under each degraded mode (same questions).  
3) Run `test/run_deepeval.py` for each CSV and report score ratios.

### 1.3 Redundancy Coverage

Define “critical components” for this repo:

- **Neo4j**
- **FAISS vector store**
- **Gemini API access**
- **Global registry** (`knowledge_graph_outputs/global_entity_registry.json`)

Define “fallback exists” when a path continues to produce a usable artifact/answer:

- A6 already has **built-in fallbacks**:
  - RAG fails → still runs graph retrieval + synthesis
  - Graph fails → still runs RAG retrieval + synthesis
- A5 has a **partial fallback** (no APOC → fallback ingestion method; upsert pass may fail)
- A1 has a **built-in alternative path** (skip annotation when >5 images; also `--no-annotate`)

**Metric**

\[
\text{Redundancy Ratio} = \frac{\#\text{critical components with fallback}}{\#\text{critical components}}
\]

Report this as a simple table (component → fallback strategy → residual capability).

---

## 2) Modularity & Decoupling Metrics (translated)

### 2.1 Coupling Metrics (Ca/Ce, Instability)

Treat each top-level folder as a “module”:

- `agents/`
- `utils/`
- `knowledge_graph/`
- `test/`

**What Ca/Ce mean here**

- **Afferent coupling (Ca)** for a module: count of *other modules* that import it.
- **Efferent coupling (Ce)** for a module: count of *other modules* it imports.
- **Instability**: \(I = \frac{Ce}{Ca + Ce}\)

**How to measure (pragmatic for this repo)**

- Start with a lightweight import-scan:
  - Count imports like `from utils...`, `from knowledge_graph...`, `from agents...`
  - Treat cross-folder imports as coupling edges
- Report a table:
  - module → Ca → Ce → I → notes

**Expected observation (already visible)**

- `agents/*` depend heavily on `utils/` and `knowledge_graph/`
- `test/*` depends on `agents/6-query_agent.py` (it imports it via `importlib`)

If you want to make the architecture *look even cleaner* in a paper, consider:

- moving shared wrappers (Neo4j connect, FAISS load, embedding wrapper) into `utils/`
- keeping agent files “thin” orchestrators over `utils/` and `knowledge_graph/`

### 2.2 Change Impact Radius (CIR)

This repo is ideal for CIR because agents are separated by **artifact boundaries** (files on disk, Neo4j, FAISS).

**Metric**

\[
\text{CIR} = \frac{\#\text{files/modules touched to implement change}}{\#\text{total files/modules in system scope}}
\]

**CIR scenarios that map directly to your architecture**

- **Swap embedding model** (`google/embeddinggemma-300m` → something else)
  - Expected touch points: `agents/4-vector_store_creation_agent.py`, `agents/6-query_agent.py`, `utils/vectordb_query.py`
  - Measure: number of modules touched (good if limited)
- **Change chunking policy** (5k/2k sizes, overlap)
  - Expected touch points: `agents/2-chunker_agent.py` (and maybe evaluation docs)
  - Measure: how many downstream components required changes (ideally none; they should read JSONL the same way)
- **Swap graph backend** (Neo4j → other graph DB)
  - Expected touch points: `agents/5-jsonl_graph_ingestion_agent.py`, `agents/6-query_agent.py`, `agents/7-validator_agent.py`, `utils/neo4j_query.py`
  - Measure: this is “big” by design; CIR will be larger

**How to measure (paper-friendly)**

- Do the change on a branch and report:
  - number of files changed
  - number of top-level modules touched (`agents/`, `utils/`, `knowledge_graph/`, `test/`)
  - optionally LOC changed

---

## 3) Scalability & Efficiency Metrics (translated)

### 3.1 Throughput Elasticity

In this repo, the most defensible “load” experiments are on **Agent 6** (query-time), because it has:

- repeatable batch driver (`test/generate_test_cases.py`)
- per-step latencies already captured (`rag_latency_ms`, `graph_latency_ms`, `synthesis_latency_ms`, `total_latency_ms`)

**Metric**

\[
\text{Elasticity} = \frac{\Delta \text{Throughput (QPS)}}{\Delta \text{Resources}}
\]

**What “resources” can mean here**

- number of parallel worker processes (e.g., N Python processes)
- Neo4j instance size (if using Aura / Docker CPU limits)

**How to measure (minimal)**

- Run the same N questions with 1 worker vs k workers, measure wall-clock time and compute QPS.
- Report speedup curve and where it saturates.

### 3.2 Bottleneck Centrality

Your query path decomposes nicely:

- RAG retrieval time
- Graph retrieval time
- Synthesis time

**Metric**

\[
\text{BottleneckScore(component)} = \frac{\text{mean latency of component}}{\text{mean total latency}}
\]

**How to measure**

- Use results produced by `test/generate_test_cases.py`:
  - mean(`rag_latency_ms`), mean(`graph_latency_ms`), mean(`synthesis_latency_ms`), mean(`total_latency_ms`)
- Report a stacked-bar chart.

### 3.3 Critical Path Length

Define “component hop” as a sequential stage that must complete before downstream can run.

- **Build path** critical path (typical): A1 → A2 → A3 → A5, and separately A2 → A4
- **Query path** critical path: A6-RAG → A6-Graph → A6-Synthesis

Report:

- number of sequential stages
- which stages are parallelizable (A4 vs A5 can be parallel after A2/A3 outputs exist)

---

## 4) Maintainability & Evolvability (translated)

### 4.1 Architectural Volatility Index (AVI)

Best measured via git history (if available):

\[
AVI(component) = \frac{\#\text{commits touching component}}{\#\text{total commits}}
\]

**How to use it in this project**

- Expect higher AVI in:
  - `agents/6-query_agent.py` (iteration-heavy)
  - `knowledge_graph/prompts.py` (prompt tuning)
- Stable core should be:
  - `knowledge_graph/models.py`
  - `utils/jsonl.py` (if used as shared IO)

If you don’t have a long git history yet, replace AVI with a “design-time” proxy:

- number of parameters exposed per agent (CLI args, env vars)
- number of hard-coded constants duplicated across agents (embedding model, paths, etc.)

### 4.2 Feature Add Cost

Use PR/commit metrics:

- LOC changed
- number of modules touched
- new interfaces introduced (new JSONL fields, new node/edge types, new env vars)

Two “features” that are architecture-representative here:

- adding a new retrieval strategy to Agent 6 (e.g., sparse retriever)
- adding a new KG validation dimension in Agent 7

---

## 5) Observability & Testability (translated)

### 5.1 Component-level test coverage

Define “component has independent tests” when it has a script/harness that runs it and outputs machine-readable metrics.

Already present:

- Agent 6 has a batch harness: `test/generate_test_cases.py`
- There is a scorer: `test/run_deepeval.py`
- Agent 7 provides deterministic structural metrics (useful as “architecture health checks”)

**Metric**

\[
\text{ComponentCoverage} = \frac{\#\text{components with runnable test harness}}{\#\text{components}}
\]

Suggested minimum components list (for the denominator):

- A1, A2, A3, A4, A5, A6, A7, Neo4j, FAISS

### 5.2 Fault Detection Latency

In this repo, measure the time from fault injection to detection as:

- time until the first error log line is emitted, OR
- time until `rag_error` / `graph_error` is populated in `LAST_QUERY_TRACE` (query path)

Practical method:

- For query path: use the timestamps and latency fields already being recorded per run.
- For pipeline path: standardize logs per agent (INFO “start/end”, ERROR on failure) and measure wall-clock.

---

## 6) Architecture-specific metrics for Hybrid RAG + KG (translated)

### 6.1 Retrieval Path Diversity (RPD)

At query time you have:

- dense retrieval via FAISS (RAG)
- graph retrieval via Neo4j (KG)

So a conservative RPD is **2**.

Optionally count “degraded-mode paths” as additional *usable* strategies:

- graph-only mode
- rag-only mode

If you do that, report both:

- **RPD (full mode)**: 2
- **RPD (with degradations)**: up to 4 (depending on how you define “strategy”)

### 6.2 Knowledge Coverage Redundancy (KG vs Docs overlap)

In this repo, a defensible approximation is:

- extract entity strings from the KG (Neo4j) and from the corpus (chunks or markdown)
- compute Jaccard overlap:

\[
KCR = \frac{|KG \cap Docs|}{|KG \cup Docs|}
\]

Practical sources you already have:

- KG entities: Neo4j nodes with labels like `Concept`, `Entity`, etc.
- Docs entities: use Agent 7’s “coverage evaluator” outputs, or (cheaper) run a lightweight NER/term extractor over `chunking_outputs/*_2k.jsonl`

### 6.3 Query Routing Entropy

Define “routing” based on which retrieval paths contribute meaningful evidence on a query:

- RAG contributes if `rag_chunks_count > 0` and `rag_error` empty
- Graph contributes if `graph_entities_count > 0` and `graph_error` empty

From a batch run, compute \(p(\text{RAG-only})\), \(p(\text{Graph-only})\), \(p(\text{Both})\), \(p(\text{Neither})\), and then:

\[
H = -\sum p_i \log p_i
\]

You can compute this directly from the CSV generated by `test/generate_test_cases.py`.

---

## 7) What to report in a paper (recommended 5–7 metrics for *this* system)

If you want a tight evaluation section, pick metrics that are:

- easy to reproduce
- aligned to your claimed contributions (agentization, robustness, hybrid retrieval, validation)

Suggested set:

- **Blast Radius Ratio** (query path + build path)
- **Graceful Degradation Score (GDS)** (Graph-off vs RAG-off)
- **Redundancy Ratio** (fallback coverage)
- **Bottleneck Centrality** (latency share: RAG vs Graph vs Synthesis)
- **Critical Path Length** (build vs query)
- **Query Routing Entropy** (diversity of evidence sources)
- **Component Coverage** (observability/testability)

---

## 8) Minimal experiment protocol (copy/paste into your “Methods”)

1) **Baseline**: run batch queries with full system available (Neo4j + FAISS + Gemini key).
2) **Degraded modes**:
   - Graph-off (Neo4j unavailable)
   - RAG-off (FAISS unavailable)
3) For each mode:
   - generate a test-case CSV via `test/generate_test_cases.py`
   - run DeepEval via `test/run_deepeval.py`
   - compute GDS ratios and routing entropy from CSV
4) Run Agent 7 structural validation on the ingested graph and report deterministic structural metrics as a “health check” baseline.

---

## 9) What we can do **without executing the pipeline** (repo scan + basic info)

This section answers: which tests can be completed by static repo scanning or by you providing basic environment info, without running Agent code / Neo4j / DeepEval. We should do these first because they’re fast and unblock “paper-ready” architecture claims.

### 9.1 Tests doable via **repo scan only** (no execution, no extra info)

**Very fast (5–20 min)**

- **Architecture inventory + component boundaries**
  - **Output**: build-time vs run-time components; stores/artifacts; dependency DAG
  - **Inputs**: folder structure + agent scripts + existing docs
- **Critical Path Length (build vs query)**
  - **Output**: sequential stages + parallelizable stages (notably A4 vs A5 after A2/A3)
  - **Inputs**: agent responsibilities + artifact directories
- **Redundancy Coverage (fallback table)**
  - **Output**: component → fallback strategy → residual capability
  - **Inputs**: code-path inspection (Agent 6 continues on FAISS/Neo4j failure; Agent 5 APOC fallback; Agent 1 skip-annotation path)
- **Retrieval Path Diversity (RPD)**
  - **Output**: RPD(full)=2 (RAG + KG); optionally list degraded-mode “paths”
  - **Inputs**: architecture definition (Agent 6 uses FAISS + Neo4j)
- **Observability/Testability coverage (qualitative baseline)**
  - **Output**: which components already output machine-readable traces/metrics (Agent 6 traces + `test/` harnesses; Agent 7 deterministic structural checks)
  - **Inputs**: presence of harness scripts/logs + known output formats
- **Architecture hygiene / “reality checks”**
  - **Output**: doc/code mismatches that harm reproducibility
  - **Inputs**: repo scan (e.g., `main_pipeline.py` referenced but missing; `agents/QUERY_AGENT_BEHAVIOR.md` referenced but missing)

**Medium (30–90 min)**

- **Coupling Metrics (Ca/Ce) + Instability**
  - **Output**: module → Ca/Ce/I plus “hot edges”
  - **Inputs**: static import scan across `agents/`, `utils/`, `knowledge_graph/`, `test/`
  - **How**:
    - manual sampling (fast, less precise), or
    - write a tiny static “import scanner” (still no pipeline execution; just reads `.py` files)
- **Design-time maintainability proxies (no git history needed)**
  - **Output**: duplication of hard-coded constants; config surface area per agent (CLI args + env vars)
  - **Inputs**: static code scan

### 9.2 Tests doable with **basic info from you** (still no execution)

These don’t measure outcomes but allow us to finalize experiment design and expected results tables.

- **Fault-injection matrix tailoring**
  - **Need from you**:
    - Neo4j: local vs Aura, APOC enabled?, full-text `entityIndex` exists?
    - expected corpus scale: #PDFs, avg pages, typical image count
    - typical hardware (CPU/RAM) for throughput claims (even if not executed yet)
  - **Output**: paper-ready fault matrix with realistic assumptions and acceptance criteria.

### 9.3 Tests that **require executing something** (defer until after the above)

- **Graceful Degradation Score (GDS)**: requires running `test/generate_test_cases.py` + `test/run_deepeval.py` in baseline and degraded modes.
- **Bottleneck Centrality**: requires measured latencies across a batch run.
- **Query Routing Entropy**: requires CSV outputs from batch runs.
- **Throughput elasticity**: requires timed runs with 1 vs k workers.
- **AVI via git history**: requires git/commit analysis (and enough history).

### 9.4 “Do these first” priority order (non-execution)

1) **Architecture inventory + boundaries** + **Critical Path Length**  
2) **Redundancy Coverage** (fallback table)  
3) **Observability/Testability coverage baseline**  
4) **Coupling/Instability** (static import scan)  
5) **Design-time maintainability proxies** (dup constants + config surface)
