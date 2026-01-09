# Validator Agent Documentation

## Overview

The Validator Agent (`agents/7-validator_agent.py`) is a comprehensive knowledge graph validation system that evaluates knowledge graphs across four key dimensions:

1. **Document Coverage** - Measures how much of the source content is represented in the graph (LLM-based)
2. **Extraction Faithfulness** - Verifies whether nodes/edges are grounded in source text (LLM-based)
3. **Graph Structural Quality** - Checks if the KG is well-formed (deterministic, no LLM required)
4. **Semantic Plausibility** - Assesses whether relations make semantic sense (LLM-based)

The agent uses **Google Gemini API** (`gemini-2.5-flash`) for LLM-based evaluations and **Neo4j** queries for structural analysis.

---

## Architecture

### Main Components

```
ValidatorAgent (Main Coordinator)
├── StructuralEvaluator (Deterministic)
├── CoverageEvaluator (LLM-based)
├── FaithfulnessEvaluator (LLM-based)
├── SemanticEvaluator (LLM-based)
└── ReportAggregator (Output formatting)
```

### Class Hierarchy

- **`ValidatorAgent`**: Main orchestrator that coordinates all evaluators
- **`StructuralEvaluator`**: Computes deterministic structural metrics
- **`CoverageEvaluator`**: Measures entity/relation coverage using LLM
- **`FaithfulnessEvaluator`**: Checks grounding of nodes/relations using LLM
- **`SemanticEvaluator`**: Validates semantic plausibility of relations using LLM
- **`ReportAggregator`**: Combines all metrics into a structured report

---

## Evaluation Dimensions

### 1. Structural Quality (Deterministic)

**Purpose**: Assess the structural integrity of the knowledge graph without requiring LLM calls.

**Metrics Computed**:

**Core Metrics**:
- **Orphan Ratio**: Percentage of nodes with degree = 0 (no connections)
  - Formula: `orphan_count / total_nodes`
  - Lower is better (0.0 = all nodes connected)

- **Component Fragmentation**: Ratio of connected components to total nodes
  - Formula: `component_count / total_nodes`
  - Lower is better (0.0005 = mostly one large component)

- **Type Violation Rate**: Rate of nodes/relationships with missing or invalid identifiers
  - Checks for:
    - Nodes without `id` property
    - Relationships with missing source/target `id` properties
  - Formula: `violations / total_relationships`
  - Lower is better (0.0 = no violations)

**Additional Quantitative Metrics** (computed automatically):
- **avg_node_degree**: Average number of connections per connected node
- **max_node_degree**: Maximum number of connections for any node
- **min_node_degree**: Minimum number of connections for any connected node
- **relationship_density**: Ratio of actual relationships to possible relationships (n × (n-1) for directed graph)
- **unique_entity_types**: Count of distinct entity types in the graph
- **unique_relationship_types**: Count of distinct relationship types in the graph
- **avg_relationships_per_node**: Average number of relationships per node
- **top_entity_types**: Top 5 most common entity types with counts (array of `{type, count}`)
- **top_relationship_types**: Top 5 most common relationship types with counts (array of `{type, count}`)
- **highest_degree_nodes**: Top 5 nodes with highest degree (array of `{id, type, degree}`)
- **connectivity_ratio**: Ratio of connected nodes to total nodes
- **document_node_count**: Total number of Document nodes in the graph
- **avg_entities_per_document**: Average number of entities linked to each Document node
- **bidirectional_relationship_ratio**: Ratio of bidirectional relationships (where reverse relation exists)

**Implementation**: Uses Neo4j Cypher queries only, no LLM required.

---

### 2. Document Coverage (LLM-based)

**Purpose**: Measures how well the knowledge graph represents the entities and relationships mentioned in the source documents.

**How It Works**:

1. **Sample chunks** from the chunking outputs (default: 5 chunks)
2. **Extract entities/relations** from each chunk using LLM (Gemini)
3. **Query graph** to find entities/relations extracted for that chunk
4. **Compare** LLM-extracted vs. graph-extracted entities/relations
5. **Calculate coverage** as intersection / LLM-extracted

**Metrics Computed**:

- **Entity Coverage**: Percentage of entities mentioned in text that are in the graph
  - Formula: `intersection(mentioned_entities, extracted_entities) / mentioned_entities`
  - Range: 0.0 to 1.0 (higher is better)

- **Relation Coverage**: Percentage of relationships mentioned in text that are in the graph
  - Formula: `intersection(mentioned_relations, extracted_relations) / mentioned_relations`
  - Range: 0.0 to 1.0 (higher is better)

**Key Methods**:

- `_extract_entities_and_relations_from_chunk()`: Uses LLM to extract both entities and relations from chunk text in a **single call** (optimization: reduces LLM requests by 50% compared to separate calls)
- `_get_extracted_entities_for_chunk()`: Queries Neo4j to find entities linked to a chunk
- `_get_extracted_relations_for_chunk()`: Queries Neo4j to find relations linked to a chunk

**Optimization**: The coverage evaluator uses a combined extraction method that extracts both entities and relationships in a single LLM call per chunk, significantly reducing API costs and execution time.

**Chunk Matching Strategy**:

The agent uses flexible matching to handle different file naming conventions:

1. **Exact match**: Try matching full `chunk_id`
2. **Suffix match**: Extract numeric suffix (e.g., `_0`) and match on that
3. **Part after `::`**: If chunk_id contains `::`, match on the part after it
4. **Numeric suffix only**: Fallback to matching just the numeric suffix
5. **Node properties**: Check `source_id` property on nodes directly

This handles cases where chunk files have different names than graph files (e.g., `attention_functional_roles_raw_chunks_2k_0` vs. `attention_functional_roles_raw_with_image_ids_with_captions_chunks_5k.jsonl::attention_functional_roles_raw_with_image_ids_with_captions_chunks_5k_0`).

---

### 3. Extraction Faithfulness (LLM-based)

**Purpose**: Verifies that nodes and relationships in the graph are actually grounded in (supported by) the source text.

**How It Works**:

1. **Sample nodes/relations** from the graph (default: 25 each)
2. **Find source chunk** for each node/relation
3. **Use LLM** to check if the node/relation is explicitly stated in the chunk text
4. **Calculate faithfulness** as percentage of grounded nodes/relations

**Metrics Computed**:

- **Node Faithfulness**: Percentage of nodes that are grounded in source text
  - Formula: `grounded_nodes / checked_nodes`
  - Range: 0.0 to 1.0 (higher is better)
  - Note: Nodes without source chunks are excluded from calculation

- **Relation Faithfulness**: Percentage of relations that are grounded in source text
  - Formula: `grounded_relations / checked_relations`
  - Range: 0.0 to 1.0 (higher is better)
  - Note: Relations without source chunks are excluded from calculation

**Key Methods**:

- `_get_source_chunk_for_node()`: Finds the source chunk text for a node
- `_get_source_chunk_for_relation()`: Finds the source chunk text for a relation
- `_check_node_grounding()`: Uses LLM to verify if a node is grounded in text
- `_check_relation_grounding()`: Uses LLM to verify if a relation is grounded in text

**Source Chunk Lookup Strategy**:

For **nodes**:
1. Find `Document` node connected via `:MENTIONS` relationship
2. Extract `source_id` from Document node
3. Match `chunk_id` in chunk files using multiple strategies
4. Return chunk content

For **relations**:
1. **Strategy 1**: Find Document that mentions both source and target (same chunk)
2. **Strategy 2**: Find Document that mentions source node (handles cross-document relations)
3. **Strategy 3**: Find Document that mentions target node (fallback)
4. **Strategy 4**: Check node `source_id` properties directly
5. Extract chunk using same matching strategies as nodes

**Special Handling**:
- Image entities (`img_*`) are filtered out from relation sampling (they use markdown sources, not chunks)
- For image relations that slip through, the agent tries to use the target's chunk instead

---

### 4. Semantic Plausibility (LLM-based)

**Purpose**: Validates whether relationships in the graph make semantic sense, regardless of whether they're explicitly stated in text.

**How It Works**:

1. **Sample relations** from the graph (default: 50)
2. **Use LLM** to check if the relation type makes sense between the source and target entities
3. **Categorize** as "yes" (plausible), "no" (implausible), or "unclear"
4. **Calculate plausibility** as `yes / (yes + no)` (unclear responses excluded)

**Metrics Computed**:

- **Plausibility Score**: Percentage of relations that are semantically plausible
  - Formula: `yes_count / (yes_count + no_count)`
  - Range: 0.0 to 1.0 (higher is better)
  - Unclear responses are excluded from the denominator

**Key Methods**:

- `_check_relation_plausibility()`: Uses LLM to assess if a relation is semantically plausible

**LLM Prompt**:
The agent asks the LLM: "Is the relation (A —[R]→ B) semantically plausible?" and provides entity types and relation type for context.

---

## Usage

### Basic Usage

```bash
# Structural evaluation only (fast, no LLM calls)
uv run agents/7-validator_agent.py

# Full evaluation with all LLM-based tests
uv run agents/7-validator_agent.py --enable-llm-tests

# Custom output directory
uv run agents/7-validator_agent.py --enable-llm-tests --output-dir "custom_reports"

# Faithfulness-only testing (for debugging)
uv run agents/7-validator_agent.py --faithfulness-only --sample-size 10
```

### Command-Line Arguments

- `--enable-llm-tests`: Enable LLM-based evaluations (coverage, faithfulness, semantic)
- `--faithfulness-only`: Run only faithfulness evaluation (for debugging)
- `--sample-size`: Override default sample size for faithfulness evaluation (default: 25)
- `--output-dir`: Custom output directory for reports (default: `validation_outputs/`)

### Environment Variables

Required in `.env` file:

```env
# Neo4j Connection
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Google Gemini API (required for LLM-based tests)
GOOGLE_API_KEY=your-api-key
```

---

## Output Format

### Report Structure

The validator generates a single aggregated report: `knowledge_graph_validation_report.json`

```json
{
  "document_id": "knowledge_graph",
  "metrics": {
    "orphan_ratio": 0.0,
    "component_ratio": 0.000544069640914037,
    "type_violation_rate": 0.0,
    "relationship_density": 0.0013795734280770736,
    "unique_entity_types": 14.0,
    "unique_relationship_types": 35.0,
    "avg_relationships_per_node": 2.5342763873775844,
    "top_entity_types": [
      {"count": 586, "type": "Concept"},
      {"count": 541, "type": "Person"},
      {"count": 242, "type": "Publication"}
    ],
    "top_relationship_types": [
      {"count": 2400, "type": "MENTIONS"},
      {"count": 617, "type": "AUTHORED_BY"},
      {"count": 526, "type": "DEPICTS"}
    ],
    "document_node_count": 30.0,
    "avg_entities_per_document": 92.30769230769232,
    "bidirectional_relationship_ratio": 0.0047230571060541005,
    "entity_coverage": 0.750,
    "relation_coverage": 0.650,
    "node_faithfulness": 0.375,
    "relation_faithfulness": 0.053,
    "semantic_plausibility": 0.850
  },
  "notes": [
    "High orphan ratio detected - many nodes are disconnected"
  ]
}
```

**Note**: All metrics are flattened into a single `metrics` dictionary. Metrics from structural evaluation are always present, while LLM-based metrics (coverage, faithfulness, semantic) are only included when `--enable-llm-tests` is used.

### Faithfulness-Only Report

When using `--faithfulness-only`, a separate report is generated: `faithfulness_only_report.json`

```json
{
  "document_id": "knowledge_graph",
  "timestamp": "2026-01-03 19:30:00",
  "faithfulness": {
    "node_faithfulness": 0.375,
    "relation_faithfulness": 0.053
  }
}
```

---

## Technical Details

### Neo4j Graph Structure

The validator expects the following Neo4j structure:

- **Document Nodes**: `(d:Document)` with `source_id` property
  - Format: `"filename::chunk_id"` (e.g., `"attention_functional_roles_raw_with_image_ids_with_captions_chunks_5k.jsonl::attention_functional_roles_raw_with_image_ids_with_captions_chunks_5k_0"`)
  
- **Entity Nodes**: Any node type with `id` property
  - Linked to Document nodes via `:MENTIONS` relationship
  
- **Relationships**: Any relationship type between nodes with `id` properties

### Chunk File Structure

Chunk files are expected in `chunking_outputs/` directory as JSONL format:

```json
{
  "id": "attention_functional_roles_raw_chunks_2k_0",
  "content": "The actual chunk text content...",
  "metadata": {...},
  "chunk_index": "0",
  "token_size_config": {...}
}
```

### LLM Integration

**Model**: Google Gemini `gemini-2.5-flash`

**Structured Outputs**: The agent uses Gemini's `response_schema` feature with manually constructed `genai.types.Schema` objects to ensure structured JSON responses. The agent defines Pydantic models for validation and converts them to Gemini Schema format:

- **`EntityAndRelationList`**: Combined model for extracting entities and relationships from chunks
  - Fields: `entities` (List[str]), `relationships` (List[Dict[str, str]])
  - Used by: CoverageEvaluator
  
- **`GroundingCheck`**: Model for checking if entities/relations are grounded in text
  - Fields: `grounded` (bool), `evidence` (str)
  - Used by: FaithfulnessEvaluator
  
- **`PlausibilityCheck`**: Model for semantic plausibility assessment
  - Fields: `plausible` (str: "yes"/"no"/"unclear"), `reasoning` (Optional[str])
  - Used by: SemanticEvaluator

**Schema Conversion**: Helper functions (`get_entity_and_relation_list_schema()`, `get_grounding_check_schema()`, `get_plausibility_check_schema()`) convert Pydantic models to Gemini's Schema format for structured output generation.

**Rate Limiting**: 
- 0.2 second delay between LLM requests (`LLM_REQUEST_DELAY = 0.2`)
- Prevents API rate limit issues while maintaining reasonable throughput

**Error Handling**:
- Failed LLM calls return default values (e.g., `False` for grounding, `"unclear"` for plausibility, empty sets for extraction)
- Invalid JSON responses are logged with warnings and execution continues
- Rate limiting delays are still applied even on errors to prevent throttling

### Matching Strategies

The validator uses sophisticated matching strategies to handle various naming conventions:

1. **Exact Match**: Direct string comparison
2. **Suffix Match**: Extract numeric suffix (e.g., `_0`) and match
3. **Contains Match**: Check if chunk_id is contained in source_id
4. **Part After `::`**: Extract and match part after `::` separator
5. **Numeric Suffix Only**: Match just the numeric suffix as fallback
6. **Node Properties**: Check `source_id` property on nodes directly

This flexibility handles cases where:
- Chunk files have different names than graph files
- File naming conventions differ (e.g., `_chunks_2k` vs. `_chunks_5k`)
- Source IDs use different formats

---

## Performance Considerations

### Sample Sizes

Default sample sizes are optimized for balance between accuracy and performance:

- **Coverage**: 5 chunks (configurable in code)
- **Faithfulness**: 25 nodes, 25 relations (configurable via `--sample-size`)
- **Semantic**: 50 relations (configurable in code)

### LLM Costs

LLM-based evaluations make multiple API calls:
- **Coverage**: ~5 calls (one per sampled chunk, extracts both entities and relations in single call)
- **Faithfulness**: ~50 calls (25 nodes + 25 relations)
- **Semantic**: ~50 calls (one per sampled relation)

**Total**: ~105 LLM calls per full validation run

**Optimization Note**: The coverage evaluator uses a combined extraction method that extracts both entities and relationships in a single LLM call per chunk, reducing what would have been ~10 calls (5 chunks × 2 extractions) to just ~5 calls.

### Execution Time

- **Structural only**: < 1 second (includes all additional quantitative metrics)
- **Full validation**: ~3-7 minutes (depending on LLM response times and rate limiting)

---

## Troubleshooting

### Common Issues

1. **"0.0 coverage/faithfulness scores"**
   - **Cause**: Chunk ID matching failed
   - **Solution**: Check that chunk files exist and naming matches graph source_ids
   - **Debug**: Use `--faithfulness-only` with smaller sample size to test

2. **"No source chunks available" warnings**
   - **Cause**: Nodes/relations not linked to Document nodes or chunk files missing
   - **Solution**: Verify Neo4j has `:MENTIONS` relationships and chunk files exist

3. **"GDS library not available" warnings**
   - **Cause**: Neo4j Graph Data Science library not installed
   - **Solution**: Not required - agent uses simple estimation method instead

4. **Low faithfulness scores**
   - **Cause**: Relations may genuinely not be grounded, or LLM is strict
   - **Solution**: Check sample relations manually, adjust LLM prompts if needed

### Debug Mode

Enable debug logging by uncommenting in code:

```python
# logger.setLevel(logging.DEBUG)
```

This will show detailed logs about:
- Chunk ID matching attempts
- Source chunk lookup strategies
- LLM request/response details

---

## Future Improvements

Potential enhancements:

1. **Parallel LLM calls**: Process multiple chunks/relations concurrently
2. **Caching**: Cache LLM responses for identical chunks/relations
3. **Configurable thresholds**: Allow users to set acceptable metric thresholds
4. **Detailed reports**: Include which specific nodes/relations failed validation
5. **Batch processing**: Validate multiple graphs in one run
6. **Visualization**: Generate visual reports with charts/graphs

---

## References

- **Code**: `agents/7-validator_agent.py`
- **Output**: `validation_outputs/knowledge_graph_validation_report.json`
- **Chunks**: `chunking_outputs/*.jsonl`
- **Graph Outputs**: `knowledge_graph_outputs/*.jsonl`
- **Neo4j Documentation**: [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graphs/neo4j)

---

## Summary

The Validator Agent provides comprehensive validation of knowledge graphs across structural, coverage, faithfulness, and semantic dimensions. It uses a combination of deterministic Neo4j queries and LLM-based evaluations to assess graph quality, with flexible matching strategies to handle various data formats and naming conventions.





