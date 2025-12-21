# Agent 6 — Query Agent (Current Behavior)

This file documents the **current runtime behavior** of `agents/6-query_agent.py`.

**Maintenance rule:** whenever you change `agents/6-query_agent.py`, update this doc in the same commit/PR.

---
## `agents/6-query_agent.py` — Hybrid Query Agent (RAG + Neo4j KG)

### Purpose

This agent answers natural-language questions by combining:

- **Dense text retrieval (RAG)** from a local FAISS index built over 2k-token chunks
- **Graph retrieval** from Neo4j using entity grounding + constrained Cypher generation + deterministic fallbacks

It then synthesizes a final response using a Gemini chat model, with explicit instructions to:

- use both contexts,
- prefer the graph for relational/structural facts,
- prefer retrieved text for detailed explanations,
- and avoid hallucinating when evidence is missing.

### How to run

From repo root:

- Ask a question:
  - `uv run agents/6-query_agent.py "how does multi-head attention differ from single-head attention?"`

### Canonical behavior doc

The file `agents/QUERY_AGENT_BEHAVIOR.md` is the authoritative, detailed runtime specification for this agent.

**Maintenance rule**: whenever `agents/6-query_agent.py` changes, update `agents/QUERY_AGENT_BEHAVIOR.md` in the same commit/PR.

### Where it sits in the pipeline

Upstream requirements:

- `vector_store_outputs/index` exists (built by `agents/4-vector_store_creation_agent.py`)
- Neo4j is populated with KG outputs (ingested using `agents/5-jsonl_graph_ingestion_agent.py` or equivalent)

Downstream:

- Produces final answers and evidence traces (logged to `logs/query_agent_logs.txt`).

### Inputs

CLI positional arg:

- `query` (optional, defaults to `"what is attention?"`)

### Outputs

- Prints `=== Final Answer ===` and the synthesized response.
- Logs detailed steps to:
  - stdout
  - `logs/query_agent_logs.txt`

### Key components (high-level)

- **RAG retrieval**:
  - loads FAISS index from `vector_store_outputs/index`
  - uses EmbeddingGemma embeddings via SentenceTransformers (`google/embeddinggemma-300m`)
  - expands the user query into subqueries via `utils.rag_rephrase.generate_rag_subqueries`
  - retrieves `k=3` per subquery, then de-duplicates chunks

- **Graph retrieval**:
  - connects to Neo4j via `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
  - grounds entities using a full-text index named `entityIndex`
  - generates Cypher via `GraphCypherQAChain` with a constrained prompt that forbids inventing entities/relationship types
  - falls back to deterministic evidence queries when the LLM query yields empty context
  - optionally appends:
    - `Images (from graph)` including image paths
    - `Chunks (...)` sections resolving chunk ids back to `chunking_outputs/*.jsonl`

### Reproducibility notes (for research writing)

- Retrieval outputs depend on the FAISS index, embedding model version, and subquery generator behavior.
- Graph answers depend on Neo4j content, presence of `entityIndex`, and the constrained Cypher generator.
- Final synthesis uses `gemini-2.5-flash` with `temperature=0`; the agent also performs deterministic post-processing to enforce evidence blocks when present in graph context.

### Paper-ready “Method” description (suggested wording)

We answer queries using a hybrid evidence model: dense retrieval over chunked document text provides explanatory context, while a Neo4j knowledge graph provides structured relational evidence. The system grounds entity mentions using a graph full-text index, generates Cypher queries under strict constraints to prevent hallucinated entities, and falls back to deterministic evidence queries when direct relations are absent. A final LLM synthesizer combines retrieved text and graph evidence into a single answer, explicitly reporting image and chunk provenance when available.

## What it does (high-level)

Given a natural-language question, the Query Agent:

1. **Retrieves text evidence** from a local FAISS vector store (RAG).
2. **Retrieves graph evidence** from Neo4j (entity grounding → LLM Cypher generation → robust fallbacks).
3. **Synthesizes a final answer** using both contexts with a Google Gemini chat model.

It logs all major steps to **stdout** and to `logs/query_agent_logs.txt`.

---

## How to run

From repo root:

```bash
uv run agents/6-query_agent.py "how does attention mechanism relate to multi head attention?"
```

If no query is provided, it defaults to:

- `what is attention?`

---

## Inputs and outputs

- **Input**: one CLI positional string argument, `query`.
- **Output**:
  - Prints `=== Final Answer ===` + the final synthesized answer to stdout.
  - Writes detailed logs to `logs/query_agent_logs.txt`.
  - Appends a structured per-query JSON trace line to `logs/query_traces.jsonl`.

---

## Environment variables / external dependencies

The agent calls `load_dotenv()` at import time, so it expects a `.env` (or real env vars) to be present.

### Required (graph retrieval)
- **`NEO4J_URI`**
- **`NEO4J_USERNAME`**
- **`NEO4J_PASSWORD`**

### Required (LLMs)
- **Google GenAI key**: used via `langchain_google_genai.ChatGoogleGenerativeAI`.
  - Typically `GOOGLE_API_KEY` (depends on your local setup).
- **Groq key**: used by `utils/rag_rephrase.generate_rag_subqueries()` (depends on that module).

### Required (local artifacts)
- `vector_store_outputs/index` must exist (FAISS store on disk).

---

## Related: why some node metadata might (not) appear in Neo4j

Your KG extraction JSONL uses `knowledge_graph.models.Node.metadata` as a list of `{key, value}` items.
For Neo4j to show these as node properties, the ingestion step must map them into LangChain node `properties`,
so they become stored as Neo4j properties.

Example input node:

- `{"id":"img_...", "type":"Image", "metadata":[{"key":"source_path","value":"...png"}]}`

Expected Neo4j node:

- `(:Image {id:"img_...", source_path:"...png"})`

---

## Step-by-step behavior

## 1) Startup + logging

- Adds project root to `sys.path`.
- Configures Python logging to:
  - stream to stdout
  - write to `logs/query_agent_logs.txt`

---

## 2) Vector store initialization

On startup (`main()`), it eagerly tries to load the FAISS store:

- Embeddings model: `google/embeddinggemma-300m` via `SentenceTransformer`
- Loads FAISS from: `vector_store_outputs/index`
- If missing or load fails, it logs a warning and continues (RAG tool will then return an error string).

---

## 3) RAG retrieval (`rag_retrieval_tool`)

Called **every run** (even if graph retrieval will fail).

Behavior:

- Generates subqueries via `utils.rag_rephrase.generate_rag_subqueries(query)`.
  - On failure, falls back to `[query]`.
- For each subquery:
  - retrieves `k=3` chunks from FAISS
  - de-duplicates by `(source_file, chunk_id, page_content)`
- Returns a string concatenation of:
  - `Content: <chunk_text>\nSource: <source_file>`

Logging:

- Logs the subqueries and the metadata of unique docs (does **not** log full chunk contents).
- Logs a summary: `num_chunks` and `total_chars`.

---

## 4) Graph retrieval (`graph_retrieval_tool`)

Called **every run** after RAG.

### 4.1 Neo4j connection + schema

- Connects using `Neo4jGraph(url, username, password)`.
- Reads:
  - `graph.schema`
  - relationship types via `CALL db.relationshipTypes() ...`

### 4.2 Entity grounding (full-text)

- Uses a Neo4j full-text index named **`entityIndex`**:
  - `CALL db.index.fulltext.queryNodes("entityIndex", $query) ...`
- Extracts a usable name with:
  - `coalesce(node.id, node.text, node.name) AS name`
- Filters out any rows where `name` is missing.

Output to Cypher generator:

- Builds `entity_str` formatted as:
  - `- <name> (<labels...>)`

### 4.3 Cypher hints to avoid “semantic edge guessing”

If the top two grounded **Concepts** have **no direct edge** in Neo4j, it injects `cypher_hints` encouraging:

- co-mention queries through `(:Document)-[:MENTIONS]->(:Concept)`
- or `shortestPath((a)-[*..4]-(b))`

### 4.4 Cypher generation (LLM)

Uses `GraphCypherQAChain.from_llm(...)` with:

- `cypher_prompt=CYPHER_PROMPT` (constrained prompt)
- `return_intermediate_steps=True`
- `verbose=True`
- `allow_dangerous_requests=True`

The prompt provides:

- schema
- grounded entities list
- allowed relationship types
- cypher hints

### 4.5 Evidence-first return behavior

If Neo4j returns **raw rows** in `intermediate_steps[1]["context"]`, the tool returns:

- `Raw graph rows:\n{...rows...}`

This prevents losing signal when the chain’s natural-language summary says “I don’t know” despite having rows.

### 4.6 Sanitization + fallback retrieval (robustness)

If raw rows are empty (or the chain answer is “I don’t know”), it attempts:

- **Sanitization pass**: rewrite relationship-type filters in the generated Cypher to remove invalid types (only applies when some types are not in the allowed set).
- **Deterministic fallbacks** (if still empty):
  - Direct Concept–Concept edges: `(a:Concept {id:$a})-[r]-(b:Concept {id:$b})`
  - Co-mention evidence through documents:
    - `(a)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(b)`
  - Shortest path (≤ 4 hops) between top concepts
  - Neighborhood expansion around the top concept

Returned fallback format:

- `Fallback graph context (LLM query returned empty rows): ...`

---

### 4.7 Image path reporting (Image nodes)

If the graph retrieval finds `:Image` nodes connected to the grounded concepts (e.g., via `DEPICTS` or document co-mentions),
it appends an explicit section to the graph context:

- `Images (from graph):`
  - each line includes **image id** and **image path** (from `source_path` or `path`)

The final answer synthesizer is instructed to always include a dedicated **"Image paths"** block in the final answer whenever
an `Images (from graph)` section is present.

---

### 4.8 Chunk reporting (Document.source_id → chunking_outputs)

When graph retrieval returns (or can infer) a link to a chunk JSONL entry, the agent resolves it to an **exact chunk** from
`chunking_outputs/<chunk_filename>`.

Two cases:

1. **Direct resolution** (best): a `Document.source_id` appears in results in the format:
   - `<chunk_filename>::<chunk_entry_id>`
   - Example: `attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl::attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k_1`
   The agent loads that JSONL file and finds the matching `{"id": "<chunk_entry_id>", ...}` line.

2. **Inference** (fallback): when the graph returns a *markdown* `Document` (e.g. `source_type=markdown`) that contains
   `derived_from_chunk_file`, the agent scans that chunk file and selects a small number of chunks whose `content`
   mentions the grounded entity names.

The graph context will include one of these sections:

- `Chunks (resolved from Document.source_id): ...`
- `Chunks (inferred from derived_from_chunk_file): ...`

The final answer is post-processed to include a **"Chunks used"** block listing the exact chunk `source_id` values whenever
one of these sections is present (and to avoid hallucinating chunk usage when not present).

---

## 5) Final answer synthesis

After both tools return strings:

- Builds a `synthesis_prompt` containing:
  - original user question
  - RAG text context
  - graph context
  - instructions: use both; prefer graph for relations; avoid hallucination
- Calls `llm_for_tools = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)`
- Prints the final answer.

---

## Logging behavior (what you should expect in logs)

- Query start + subqueries + retrieval stats
- Neo4j connection
- Grounded entities list
- Generated Cypher
- Raw Neo4j context rows (if available)
- Fallback query execution notice (only when needed)
- Final answer text
- Structured query trace emitted after the answer (see below)

---

## Structured per-query trace logging (JSON)

After printing the final answer, the agent emits a **structured JSON record** and persists it for later analysis.

### Where it is written

- **File (JSONL)**: `logs/query_traces.jsonl`
  - One JSON object per line, appended after each run.
- **Stdout**: printed after the final answer under:
  - `=== Query Trace (JSON) ===`
- **Text log**: also logged as a single-line `QUERY_TRACE_JSON ...` entry in `logs/query_agent_logs.txt`.

### What is captured per query

The per-query JSON record includes:

- **User question**: `user_question`
- **Rewritten / decomposed queries**: `rewritten_or_decomposed_queries`
- **Retrieved chunks**: `retrieved_chunks`
  - includes `chunk_id`, `source_file`, chunk `metadata`, and chunk `text`
- **Retrieved graph subgraph**: `retrieved_graph_subgraph`
  - `nodes`: list of `{id, labels, properties}`
  - `edges`: list of `{source, type, target, properties}`
- **Final answer**: `final_answer`
- **Token usage**: `token_usage` (best-effort; may be null depending on provider metadata availability)
- **Latency**: `latency_ms`
  - `rag`, `graph`, `synthesis`, `total`

For debugging, the record also includes an `internal` block that contains the raw per-tool trace objects (`rag` and `graph`) used to build the top-level summary fields.

---

## Known constraints / assumptions

- Neo4j must have a full-text index named **`entityIndex`**.
- The KG may connect concepts mainly via **document co-mentions**, so many “semantic” edges between concepts may not exist; fallbacks handle this.
- `allow_dangerous_requests=True` is enabled for the Cypher chain (powerful but risky—keep Neo4j credentials scoped).


