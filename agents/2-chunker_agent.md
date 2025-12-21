## `agents/2-chunker_agent.py` — Chunker Agent

### Purpose

This agent transforms a single Markdown document in `markdown_outputs/` into **two JSONL chunk corpora**:

- **5k-token chunks**: intended for **knowledge graph construction** (more context per chunk)
- **2k-token chunks**: intended for **retrieval (RAG / similarity search)** (finer granularity)

It exposes one tool: `chunk_markdown_tool(markdown_filename)`.

### How to run

From repo root:

- Chunk a specific markdown file (by filename):
  - `uv run agents/2-chunker_agent.py sliding_window_attention_raw_with_image_ids_with_captions.md`

### Where it sits in the pipeline

Upstream:

- Consumes Markdown produced by `agents/1-pdf_processor_agent.py` (raw or annotated).

Downstream:

- `*_chunks_5k.jsonl` feeds `agents/3-graph_data_extractor_agent.py` (graph extraction).
- `*_chunks_2k.jsonl` feeds `agents/4-vector_store_creation_agent.py` (FAISS vector store).

### Inputs

- **Tool input**: `markdown_filename` (string filename, not a full path)
  - Must exist at: `markdown_outputs/<markdown_filename>`

CLI usage (current implementation):

- If an argument is provided, the script uses `Path(sys.argv[1]).name` and passes that filename to the tool.
- If no argument is provided, it defaults to:
  - `sliding_window_attention_raw_with_image_ids_with_captions.md`

### Outputs / artifacts

For `markdown_outputs/<stem>.md`, the tool writes into `chunking_outputs/`:

- `chunking_outputs/<stem>_chunks_5k.jsonl`
- `chunking_outputs/<stem>_chunks_2k.jsonl`

### Chunking method

Uses `langchain_text_splitters.RecursiveCharacterTextSplitter.from_tiktoken_encoder()`:

- **5k config**:
  - `chunk_size=5000`
  - `chunk_overlap=500`
  - suffix `_chunks_5k`
- **2k config**:
  - `chunk_size=2000`
  - `chunk_overlap=200`
  - suffix `_chunks_2k`

This is token-aware chunking via the tiktoken encoder (best-effort token approximation).

### JSONL record schema (what each line contains)

Each output file is JSONL. Each line is one chunk record:

- **`id`**: `f"{stem}{suffix}_{i}"`
- **`content`**: the chunk text
- **`metadata`**: dict; at minimum `{ "source": "<markdown_filename>" }`
- **`chunk_index`**: integer
- **`token_size_config`**: the configured chunk size (2000 or 5000)

### Special features / noteworthy behaviors

- **Dual-output design**: produces both retrieval-optimized and KG-optimized corpora in one run.
- **Traceability**: the `metadata.source` field is later relied on by the graph extractor for image entity extraction (it uses this to find the originating Markdown file).

### Failure modes

- Missing input markdown → returns a string error `"Error: Input file not found at ..."`
- Unexpected exceptions → returns `"Error during chunking: ..."` and logs stack trace

### Reproducibility notes (for research writing)

- Chunk boundaries are deterministic given identical input markdown and splitter configuration.
- Tokenization approximation depends on the underlying tiktoken encoder used by LangChain.

### Paper-ready “Method” description (suggested wording)

We transform each document into overlapping token-based chunks using a tiktoken-aware recursive character splitter. To support both structured extraction and retrieval, we generate two corpora: a coarse-grained 5k-token corpus for knowledge graph construction and a finer-grained 2k-token corpus for dense retrieval, with overlaps of 500 and 200 tokens respectively to preserve cross-chunk continuity.


