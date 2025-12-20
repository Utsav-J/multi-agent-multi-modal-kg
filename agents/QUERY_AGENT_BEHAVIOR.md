# Agent 6 — Query Agent (Current Behavior)

This file documents the **current runtime behavior** of `agents/6-query_agent.py`.

**Maintenance rule:** whenever you change `agents/6-query_agent.py`, update this doc in the same commit/PR.

---

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

---

## Known constraints / assumptions

- Neo4j must have a full-text index named **`entityIndex`**.
- The KG may connect concepts mainly via **document co-mentions**, so many “semantic” edges between concepts may not exist; fallbacks handle this.
- `allow_dangerous_requests=True` is enabled for the Cypher chain (powerful but risky—keep Neo4j credentials scoped).


