## `agents/5-jsonl_graph_ingestion_agent.py` — JSONL Graph Ingestion (Neo4j)

### Purpose

This script ingests graph extraction outputs (JSONL) into a Neo4j database using `langchain_neo4j`.

It is the “database materialization” step: it converts the project’s custom Pydantic graph schema (`knowledge_graph.models.GraphDocument`) into LangChain’s Neo4j `GraphDocument` format, then writes nodes/relationships (and source provenance) into Neo4j.

> Note: despite the filename, this is not a LangChain “agent” wrapper like the others; it is a direct ingestion script.

### How to run

From repo root:

- Run the script as-is (it ingests the single, hard-coded `input_file` in the `__main__` block):
  - `uv run agents/5-jsonl_graph_ingestion_agent.py`

If you want to ingest a different JSONL, update the `input_file = ...` assignment in the script (or import and call `load_jsonl_and_ingest(...)` from another driver).

### Where it sits in the pipeline

Upstream:

- Consumes JSONL outputs written by `agents/3-graph_data_extractor_agent.py`, e.g.:
  - `knowledge_graph_outputs/*_graph.jsonl` (text KG)
  - `knowledge_graph_outputs/*_images_graph.jsonl` (image KG)

Downstream:

- Produces a queryable Neo4j graph used by `agents/6-query_agent.py` (`graph_retrieval_tool`).

### Inputs

The main block currently hard-codes an input file path:

- `knowledge_graph_outputs/attention_is_all_you_need_raw_with_image_ids_with_captions_chunks_5k_graph.jsonl`

The core ingestion function is generic:

- `load_jsonl_and_ingest(file_path: str, graph: Neo4jGraph)`

### Outputs / side effects

- Creates / merges nodes and relationships in Neo4j.
- Adds source provenance (depends on `include_source=True`).

### Environment variables (Neo4j connection)

`connect_to_neo4j()` reads:

- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `password`)

### Data model conversion

#### Custom schema (project)

This script expects each JSONL line to contain (at minimum):

- `nodes`: list of `knowledge_graph.models.Node`
- `relationships`: list of `knowledge_graph.models.Relationship`
- `source`: `knowledge_graph.models.Document` (provenance pointer)

It tolerates extra top-level fields in JSONL lines (they are ignored for graph structure).

#### Conversion to LangChain Neo4j format

`convert_to_langchain_format(custom_graph_doc: GraphDocument) -> LCGraphDocument`:

- **Source document**
  - Creates `langchain_core.documents.Document` with:
    - `page_content = source_id` (a provenance pointer, not full text)
    - `metadata = {key:value}` derived from `Document.metadata`
    - Always includes `source_id` and `source_type` in metadata when available

- **Nodes**
  - Converts each node to `langchain_neo4j.graphs.graph_document.Node` with:
    - `id`, `type`
    - `properties` mapped from `Node.metadata` (`MetadataItem` list)

`MetadataItem` → properties mapping rules:

- Duplicate keys become lists (Neo4j supports list properties).
- Values are stored as strings (as defined in `MetadataItem`).

- **Relationships**
  - Converts each relationship to `langchain_neo4j.graphs.graph_document.Relationship`:
    - `source` and `target` as lightweight nodes (`id`, `type`)
    - `type`
    - `properties={}` (no relationship properties are written here)

### Ingestion behavior

`load_jsonl_and_ingest(...)`:

- Reads JSONL line-by-line.
- Reconstructs:
  - `Document` (provenance), including its `metadata` list if present
  - `Node` list
  - `Relationship` list (reconstructing embedded `source`/`target` node dicts)
- Converts to LangChain format and collects `LCGraphDocument` objects.
- Writes to Neo4j via:
  - `graph.add_graph_documents(lc_documents, include_source=True)`

### Special feature: node property “upsert” pass (important)

After `add_graph_documents`, the script performs an explicit property upsert pass:

- It builds a list of `{id, type, properties}` for all nodes in the batch.
- Runs a Neo4j query:
  - `CALL apoc.merge.node([row.type], {id: row.id}, row.properties, row.properties)`

Reason (as documented in code):

- `langchain_neo4j` uses `apoc.merge.node(..., onMatchProps={})` by default, which may not update properties of already-existing nodes.
- This upsert ensures metadata (e.g., `source_path` for images) becomes visible even when nodes already exist.

### Failure modes

- JSON decode errors are logged per line and do not stop the entire run.
- Missing file path prints `"File not found: ..."`
- If Neo4j is unreachable or APOC is unavailable, ingestion will fail (or the upsert step will warn).

### Reproducibility notes (for research writing)

- Ingestion is deterministic given identical JSONL inputs.
- If the same nodes are ingested multiple times, node identity is keyed on `{id}` per label/type; property update behavior depends on APOC availability and the explicit upsert step.

### Paper-ready “Method” description (suggested wording)

We persist extracted graph documents into Neo4j by converting each JSONL line (nodes, relationships, and provenance metadata) into Neo4j-compatible labeled nodes and typed edges. Node metadata is materialized as node properties, with an explicit APOC-based upsert step to ensure properties are updated when entities are re-encountered across documents or re-ingestion runs.


