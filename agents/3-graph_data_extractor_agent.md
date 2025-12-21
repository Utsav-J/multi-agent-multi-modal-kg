## `agents/3-graph_data_extractor_agent.py` — Graph Data Extractor Agent

### Purpose

This agent is responsible for generating **knowledge-graph-ready JSONL outputs** from chunked text and from image metadata embedded in Markdown.

It contains two tools:

- **`extract_image_entities_from_chunks_tool(chunks_filename, include_metadata=False)`**
  - Builds a deterministic **image subgraph** from fenced JSON blocks in the originating Markdown.
- **`extract_graph_from_chunks_tool(chunks_filename, include_metadata=False, batch_size=1, token_limit=5500)`**
  - Uses Gemini JSON-schema constrained generation to extract **nodes and relationships** from text chunks.

It also maintains a global **entity registry** to encourage stable entity naming across multiple documents.

### How to run

From repo root (typical case: run on a 5k chunk file):

- Include metadata (default behavior in this script unless `--no-metadata` is set):
  - `uv run agents/3-graph_data_extractor_agent.py attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl`
- Exclude original chunk metadata in output:
  - `uv run agents/3-graph_data_extractor_agent.py attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl --no-metadata`
- Adaptive batching token limit:
  - `uv run agents/3-graph_data_extractor_agent.py attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl --token-limit 5500`
- Fixed batch size (only used when token-limit is not set / <= 0):
  - `uv run agents/3-graph_data_extractor_agent.py attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k.jsonl --batch-size 2`

### Where it sits in the pipeline

Upstream:

- Consumes `chunking_outputs/*_chunks_5k.jsonl` produced by `agents/2-chunker_agent.py`.
- For image extraction, it also consumes the originating Markdown files referenced by chunk metadata (`metadata.source`).

Downstream:

- Produces JSONL graph documents under `knowledge_graph_outputs/` that can be ingested into Neo4j using `agents/5-jsonl_graph_ingestion_agent.py` (or equivalent ingestion).

### Inputs

#### Tool: `extract_image_entities_from_chunks_tool`

- **`chunks_filename`**: filename in `chunking_outputs/` (expects JSONL records with `metadata.source`)
- **`include_metadata`**: optional flag to include extra convenience fields in the written JSONL

#### Tool: `extract_graph_from_chunks_tool`

- **`chunks_filename`**: filename in `chunking_outputs/`
- **`include_metadata`**: if true, adds original chunk metadata into the serialized graph JSON
- **`batch_size`**: fixed-size batching only used when adaptive token limit is disabled
- **`token_limit`**: adaptive batching limit (approx); see batching behavior below

### Outputs / artifacts

All outputs are written under `knowledge_graph_outputs/`:

- **Text graph**: `knowledge_graph_outputs/<chunks_stem>_graph.jsonl`
- **Image graph**: `knowledge_graph_outputs/<chunks_stem>_images_graph.jsonl`
- **Global registry**: `knowledge_graph_outputs/global_entity_registry.json`

Each output JSONL line is a serialized `knowledge_graph.models.GraphDocument`-compatible dict:

- `nodes`: list of `{id, type, metadata?}`
- `relationships`: list of `{source:{id,type}, target:{id,type}, type}`
- `source`: a provenance pointer `Document` (see below)

### Global entity registry

- Registry path: `knowledge_graph_outputs/global_entity_registry.json`
- Stored as a JSON list of entity ids; loaded into a Python `set` at runtime.
- Updated after each processed image-markdown file and after each processed text batch.
- For text extraction, the registry is also **refreshed from disk per batch** so concurrent runs can still converge on a shared set.

This registry is used as **context** (a list of existing entity ids) during LLM extraction to improve cross-document consistency.

---

## Tool 1: `extract_image_entities_from_chunks_tool`

### Core idea

Instead of running a vision model here, this tool reads **structured image metadata** already embedded in Markdown, and turns it into deterministic graph nodes and edges.

### How it finds images

1. Loads the chunk JSONL from `chunking_outputs/<chunks_filename>`.
2. Collects unique markdown sources from each chunk record:
   - `record["metadata"]["source"]` (a markdown filename)
3. For each markdown file `markdown_outputs/<source>`:
   - Extracts fenced blocks matching ```json ... ```
   - Each block is expected to define one object of the form:
     - `"img_...": { ... }`
   - A lenient JSON loader removes trailing commas before parsing.

### Deterministic graph schema (image subgraph)

For each markdown source:

- **Nodes**
  - `Document`: `id = Path(source).stem`
  - `Image`: `id = image_id`, `type="Image"`
    - Metadata: `source_path=<image_path>` when present
  - `Section`: `id = f"{document_id}::section::{section}"`
  - `Concept`: one per string in `depicted_concepts`

- **Relationships**
  - `(Image)-[:PART_OF]->(Document)`
  - `(Image)-[:LOCATED_IN]->(Section)`
  - `(Image)-[:DEPICTS]->(Concept)` for each depicted concept

### Hard rules / filters

- Images without a `caption` are **skipped** (hard rule in code).
- `section` defaults to `"Unknown"` if missing.
- Only `image_id` values starting with `"img_"` are accepted.

### Provenance (`source` field)

Each image graph line writes a `source` Document with:

- `source_id = f"{source}::images"`
- `source_type = "markdown"`
- metadata:
  - `markdown_source=<source>`
  - `derived_from_chunk_file=<chunks_filename>`

### Output path

- `knowledge_graph_outputs/<chunks_stem>_images_graph.jsonl`

---

## Tool 2: `extract_graph_from_chunks_tool`

### Core idea

This tool performs **LLM-based structured extraction** from chunked text using Google GenAI’s JSON-schema constrained generation:

- `response_mime_type="application/json"`
- `response_schema=knowledge_graph.models.GraphDocument`

The prompt is produced by:

- `knowledge_graph.prompts.render_graph_construction_instructions(...)`

and includes:

- the chunk text (or batch-concatenated chunk text),
- the list of existing entity ids from the global registry,
- and a provenance “source document” serialized as JSON.

### Batching behavior (important)

The implementation supports:

- **Adaptive token-limit batching** (default path)
  - token estimate: `len(content)//4` (very rough)
  - when adding a chunk would exceed `effective_token_limit`, it flushes the current batch
  - if `token_limit` is not provided (0) and `batch_size==1`, it sets `effective_token_limit=5500`
- **Fixed batch-size batching**
  - only used when `token_limit<=0` and `batch_size>1`

### Provenance (`source` field)

For each processed batch, it constructs a `knowledge_graph.models.Document` pointer:

- **Single chunk**:
  - `source_id = f"{chunks_filename}::{cid}"`
    - where `cid` is the chunk `id` if present, otherwise a fallback based on chunk index
  - `source_type = "chunk"`
  - metadata includes `chunk_file`, `chunk_id`, and optionally `chunk_index`

- **Multi-chunk batch**:
  - `source_id = f"{chunks_filename}::batch::<first_chunk_id_or_unknown>"`
  - `source_type = "chunk_batch"`
  - metadata may include JSON-encoded `chunk_ids` and `chunk_indices`

The returned `GraphDocument` is then overwritten to ensure:

- `graph_doc.source = source_doc`

### Optional traceability fields (`include_metadata=True`)

If enabled, it appends to the serialized graph dict:

- `original_chunk_ids`: list
- `original_chunk_indices`: list
- `original_metadata`: list of metadata dicts per chunk

### Output path

- `knowledge_graph_outputs/<chunks_stem>_graph.jsonl`

---

## Agent orchestration (system prompt)

The LangChain agent is configured to run in **one-shot** mode and is instructed that:

- It should run **image extraction first**, then text graph extraction.
- It should not ask follow-up questions.
- It should return only final results.

### Environment / dependencies

- Loads environment variables via `dotenv.load_dotenv()`.
- Uses `google.genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))` for extraction calls.
- Also initializes `ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)`, used only to create the LangChain agent wrapper.
- Relies on:
  - `knowledge_graph.models.GraphDocument` (Pydantic schema)
  - `knowledge_graph.prompts.render_graph_construction_instructions` (prompt construction)

### Failure modes / robustness

- Missing chunk file or markdown source file → logs warning; image extraction skips missing markdown.
- JSON parsing in fenced blocks is best-effort; malformed blocks are ignored.
- Exceptions in a batch do not stop processing all batches; errors are logged per batch.
- Registry persistence is “write-through” (saved after each unit of work), which improves recoverability.

### Reproducibility notes (for research writing)

- The image subgraph is deterministic given the markdown-embedded metadata.
- The text KG extraction depends on LLM behavior; settings used:
  - `gemini-2.5-flash` with JSON schema enforcement
  - registry-provided entity ids as grounding context
- Batching affects model context windows and can change extraction results; report the batching policy used (adaptive 5500-token default vs explicit `token_limit` / `batch_size`).

### Paper-ready “Method” description (suggested wording)

We build a multimodal knowledge graph in two stages. First, we deterministically extract an image subgraph by parsing structured image metadata embedded in the document Markdown and instantiating nodes for Images, Documents, Sections, and depicted Concepts, with edges encoding containment and depiction relations. Second, we perform LLM-based structured extraction over overlapping 5k-token text chunks, using JSON-schema-constrained generation to produce typed nodes and relationships. To encourage cross-document consistency, we maintain a global entity registry that is injected as context into subsequent extraction calls.


