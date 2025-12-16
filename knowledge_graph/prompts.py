from typing import List, Optional
from langchain_community.graphs import Neo4jGraph

graph_construction_instructions = """You are an information extraction system for building a reusable scientific knowledge graph from research papers.

Your goal is to extract ONLY high-value, reusable entities and relationships.
You must avoid over-extraction and paper-local details.

====================
ENTITY SELECTION RULES
====================

Extract entities ONLY if they satisfy at least one of the following:
1. A named scientific model, method, or framework reused across papers
2. A clearly defined task or application domain
3. A published work (author + year citation)
4. A widely accepted scientific concept (field-level, not paper-specific)
5. An organization or author (for provenance)

DO NOT extract:
- Variables, symbols, or mathematical notation (e.g., q, k, a_t, φ, ωτ)
- Internal architectural components unless they are independently reusable methods
- Solver choices unless the paper’s main contribution is numerical methods
- Theorems, lemmas, corollaries, propositions, or equation numbers
- Section titles, analysis labels, or descriptive phrases
- Paper-specific reformulations unless claimed as a standalone method

====================
ABSTRACTION RULES
====================

Prefer higher-level abstractions:
- Use “Attention Mechanisms” instead of “Attention Logits”
- Use “Continuous-Time Modeling” instead of “Closed-Form ODE Solution”
- Use “Biologically Inspired Models” instead of individual gate mechanisms

Merge synonymous or closely related entities.
Avoid creating multiple nodes for minor variations.

====================
ENTITY TYPES (STRICT)
====================

Allowed entity types:
- Model
- Task
- Concept
- Algorithm (only if novel and reusable)
- Publication
- Person
- Organization

If an entity does not clearly fit one of these types, DO NOT extract it.

====================
RELATIONSHIP RULES
====================

Extract relationships only if they express:
- INTRODUCED_BY
- BUILDS_ON
- COMPARED_WITH
- APPLIED_TO
- INSPIRED_BY
- EXTENDS
- MITIGATES
- ADDRESSES
- AUTHORED_BY
- AFFILIATED_WITH

Avoid relationships describing implementation mechanics.
Favor precision over recall.
If unsure about an entity, OMIT it.

## Pre-existing Entities
The following entities have already been extracted from previous chunks. If you encounter these entities again, reuse their exact IDs and types:
{existing_entities}

---

## Output Format
- Return **only** a valid JSON object matching the GraphDocument structure:
- Example:
```json
{
  "nodes": [
    {
      "id": "Rex",
      "type": "Animal"
    },
    {
      "id": "London",
      "type": "Location"
    }
  ],
  "relationships": [
    {
      "source": {
        "id": "Rex",
        "type": "Animal"
      },
      "target": {
        "id": "London",
        "type": "Location"
      },
      "type": "LIVES_IN"
    }
  ],
}
```

Do **not** include explanations, commentary, or extra text.
If no entities or relations are found, return an empty JSON:

```json
{"nodes": [], "relationships": []}
```

# Text to Process

Here is the document chunk to extract entities and relationships from:
{chunk}
"""


def render_graph_construction_instructions(
    chunk: str, existing_entities: Optional[List[str]] = None
) -> str:
    formatted_existing_entities = "None"
    if existing_entities:
        formatted_existing_entities = ", ".join(existing_entities)

    return graph_construction_instructions.replace("{chunk}", chunk).replace(
        "{existing_entities}", formatted_existing_entities
    )


# if __name__ == "__main__":
#   print(render_graph_construction_instructions("hey"))
