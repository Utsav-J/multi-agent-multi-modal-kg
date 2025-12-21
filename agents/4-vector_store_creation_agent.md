## `agents/4-vector_store_creation_agent.py` — Vector Store Creation Agent

### Purpose

This agent creates a local dense-retrieval index for RAG by:

1. Scanning `chunking_outputs/` for all `*_2k.jsonl` chunk files
2. Embedding each chunk with **EmbeddingGemma** (via SentenceTransformers)
3. Building a **FAISS** index and saving it under `vector_store_outputs/index`

It exposes one tool: `scan_and_ingest_chunks()`.

### How to run

From repo root:

- Build/rebuild the FAISS index from all `chunking_outputs/*_2k.jsonl` files:
  - `uv run agents/4-vector_store_creation_agent.py`

### Where it sits in the pipeline

Upstream:

- Consumes the `_chunks_2k.jsonl` outputs produced by `agents/2-chunker_agent.py`.

Downstream:

- Produces the FAISS index consumed by `agents/6-query_agent.py` (`rag_retrieval_tool`).

### Inputs

Tool signature:

- `scan_and_ingest_chunks(dummy_arg: str = "")`
  - `dummy_arg` is not used; it exists only to satisfy tool signature expectations.

CLI:

- `--input` is passed to the agent as a trigger string (default: `"start chunking"`), but the tool always scans the directory regardless of content.

### Outputs / artifacts

- **FAISS index directory**: `vector_store_outputs/index`
  - Created via `FAISS.from_documents(...)` then `vector_store.save_local(...)`

### Embedding model details

Defines `EmbeddingGemmaWrapper(Embeddings)`:

- Backed by: `SentenceTransformer("google/embeddinggemma-300m")`
- Document embeddings:
  - `prompt_name="document"`
  - `normalize_embeddings=True`
- Query embeddings:
  - `prompt_name="query"`
  - `normalize_embeddings=True`

### Chunk ingestion (JSONL parsing)

For each `chunking_outputs/*_2k.jsonl` file:

- Reads JSONL lines; for each parsed object:
  - uses `content = chunk_data.get("content", "")`
  - builds a `langchain_core.documents.Document(page_content=content, metadata=...)`
- Metadata handling:
  - If `chunk_data["metadata"]` is a dict:
    - adds `chunk_id` (from `chunk_data["id"]` or fallback)
    - adds `source_file` (the chunk file name)
  - Otherwise, constructs a minimal metadata dict with those keys.

### Special features / noteworthy behaviors

- **Auto-discovery**: processes all `*_2k.jsonl` files present; no explicit file list required.
- **Graceful JSON errors**: logs a warning and continues on per-line JSON decode failures.

### Failure modes

- Missing `chunking_outputs/` directory → returns an error string.
- No matching `*_2k.jsonl` files → returns a message indicating none were found.
- If no valid chunks are found after parsing → returns `"No valid chunks found to ingest."`
- FAISS creation or save failures are caught and returned as error strings.

### Environment / dependencies

- Loads `.env` via `dotenv.load_dotenv()`.
- Uses:
  - `langchain_community.vectorstores.FAISS`
  - `sentence_transformers.SentenceTransformer`
  - `langchain_google_genai.ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)` only to wrap the tool in a LangChain agent (indexing itself is local).

### Reproducibility notes (for research writing)

- Given identical chunk JSONL inputs and identical embedding model weights/version, the FAISS index should be reproducible.
- Hardware/backends may introduce minor floating-point variance; normalization reduces but does not always eliminate drift.
- Any preprocessing that changes chunk text (e.g., Markdown cleaning) will change embeddings and retrieval.

### Paper-ready “Method” description (suggested wording)

We index the document corpus for dense retrieval by embedding 2k-token overlapping chunks using the `google/embeddinggemma-300m` sentence embedding model (document/query prompting with L2-normalized vectors) and constructing a FAISS index over the resulting embeddings. The index is persisted locally and later used to retrieve top-k chunks as textual evidence for downstream question answering.


