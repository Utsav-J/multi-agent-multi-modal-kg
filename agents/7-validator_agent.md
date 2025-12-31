# Validator Agent

The Validator Agent validates knowledge graphs across four dimensions as specified in the validator plan:

1. **Document Coverage** - How much of the source content is represented in the graph
2. **Extraction Faithfulness** - Whether nodes/edges are grounded in the text
3. **Graph Structural Quality** - Whether the KG is well-formed and usable
4. **Semantic Plausibility** - Whether relations make sense given entity types

## Current Implementation Status

### Fully Implemented (Deterministic Tests)

- **StructuralEvaluator**: Implements all three structural quality tests:
  - **Orphan Ratio**: Percentage of nodes with degree = 0
  - **Component Fragmentation**: Ratio of connected components to total nodes
  - **Type Consistency**: Rate of nodes/relationships with missing or invalid types

### Framework Provided (Requires LLM Integration)

- **CoverageEvaluator**: Framework for entity and relationship coverage metrics
- **FaithfulnessEvaluator**: Framework for node and relationship grounding checks
- **SemanticEvaluator**: Framework for semantic plausibility checks

These LLM-based evaluators return placeholder metrics and log warnings when invoked. They can be implemented by integrating an LLM (e.g., via LangChain) to perform the actual evaluations.

## Usage

### Basic Usage (Structural Tests Only)

```bash
python agents/7-validator_agent.py
```

This will:
- Connect to Neo4j (using environment variables)
- Detect all documents in the graph
- Run structural quality evaluations
- Generate validation reports in `validation_outputs/`

### Validate Specific Document

```bash
python agents/7-validator_agent.py --document-id "attention_is_all_you_need"
```

### Enable LLM-Based Tests (Placeholders)

```bash
python agents/7-validator_agent.py --enable-llm-tests
```

Note: Currently this will run placeholder evaluations. To fully enable LLM tests, you need to integrate an LLM into the CoverageEvaluator, FaithfulnessEvaluator, and SemanticEvaluator classes.

### Custom Output Directory

```bash
python agents/7-validator_agent.py --output-dir "custom_output"
```

## Output Format

The validator produces JSON reports with the following structure:

```json
{
  "document_id": "attention_is_all_you_need",
  "metrics": {
    "orphan_ratio": 0.08,
    "component_ratio": 0.03,
    "type_violation_rate": 0.02,
    "entity_coverage": 0.0,
    "relation_coverage": 0.0,
    "node_faithfulness": 0.0,
    "relation_faithfulness": 0.0,
    "semantic_plausibility": 0.0
  },
  "notes": [
    "High orphan ratio detected - many nodes are disconnected"
  ]
}
```

## Environment Variables

The agent requires the following environment variables (via `.env` file):

- `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `password`)

## Architecture

The validator follows a modular design:

```
ValidatorAgent
 ├── StructuralEvaluator      (✅ Implemented - deterministic)
 ├── CoverageEvaluator         (📋 Framework - requires LLM)
 ├── FaithfulnessEvaluator     (📋 Framework - requires LLM)
 ├── SemanticEvaluator         (📋 Framework - requires LLM)
 └── ReportAggregator          (✅ Implemented)
```

## Implementation Notes

### Structural Evaluator

The structural evaluator uses Neo4j Cypher queries to compute metrics:

- **Orphan Detection**: Uses `WHERE NOT (n)--()` to find isolated nodes
- **Component Counting**: Attempts to use GDS library first, falls back to traversal-based methods
- **Type Validation**: Checks for nodes without `id` property and relationships with invalid endpoints

### Future LLM Integration

To implement the LLM-based evaluators, you would:

1. Initialize an LLM (e.g., `ChatGoogleGenerativeAI`) in each evaluator's `__init__`
2. Sample nodes/relationships from the graph
3. Retrieve corresponding source chunks/text
4. Use the LLM to:
   - Extract entities/relations from chunks (coverage)
   - Check if graph entities are grounded in text (faithfulness)
   - Assess semantic plausibility of relations (semantic)

Example prompt structure for faithfulness checking:
```
"Is the entity '{entity_id}' explicitly stated or clearly implied in the provided text? 
Answer yes/no and quote evidence.

Text: {source_chunk_text}"
```

## Validation Philosophy

As stated in the validator plan:

> The validator is **consistency- and coverage-driven**, not truth-omniscient.

Key assumptions:
- The **source Markdown is the only ground truth**
- The validator checks:
  - *Did we extract what is present?* (coverage)
  - *Did we hallucinate what is not present?* (faithfulness)
  - *Is the graph structurally sane?* (graph quality)

This positions the validator as a **self-consistency auditor**, not a fact checker.



