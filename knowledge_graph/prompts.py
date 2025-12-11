from typing import List, Optional
from langchain_community.graphs import Neo4jGraph

graph_construction_instructions = """ # Knowledge Graph Construction Instructions

## 1. Role and Objective
You are an advanced information extraction model specializing in transforming unstructured text into structured knowledge graph data.  
Your task is to read the given text and extract all **entities (nodes)** and **relationships (edges)** relevant for constructing a knowledge graph.  
Each relationship is a **triple** in the form of:

```
Source_Node ~~ Relationship ~~ Target_Node
````

Your goal:
- Maintain **semantic accuracy**, **naming consistency**, and **clean minimalism**.
- Produce results that can be directly converted into graph database nodes and edges.
- Ensure consistent capitalization and spelling so that nodes referring to the same concept are identical.


## 2. Node and Relationship Guidelines

### 🟢 Node Rules
- **Nodes represent real-world entities or abstract concepts.**
- Label every node with a general, simple **type** (e.g., "Person", "Organization", "Location", "Event", "Concept").
- The node **id** should be the entity name or phrase as it appears in the text (clean and human-readable).
- Example: `"Elon Musk"` → `{"id": "Elon Musk", "type": "Person"}`

### 🟠 Relationship Rules
- Relationships should describe **clear, meaningful connections** (verbs or prepositions).
- Use uppercase snake case (e.g., `FOUNDED`, `BORN_IN`, `LOCATED_AT`, `WORKS_FOR`, `PART_OF`).
- Keep them **directional** — from subject (source) to object (target).
- Example: `"Elon Musk founded SpaceX"` → `{"source": "Elon Musk", "target": "SpaceX", "type": "FOUNDED"}`

---

## 3. Data Normalization and Attributes
- **No separate nodes for numeric or date values.**
  - Attach such data as node attributes.
  - Example: `"Elon Musk (born 1971)"` → `{"id": "Elon Musk", "type": "Person", "attributes": {"birthYear": 1971}}`
- **Use camelCase for property names** (e.g., `birthDate`, `foundedYear`).
- Do **not** escape quotes within values.

---

## 4. Coreference and Entity Consistency
- If an entity is referred to by pronouns or abbreviations later, always use its **most complete name**.
  - Example: “Musk” → “Elon Musk”
- The goal is a **coherent and unified graph**, not duplicate nodes.

## 5. Pre-existing Entities
The following entities have already been extracted from previous chunks. If you encounter these entities again, reuse their exact IDs and types:
{existing_entities}

---

## 6. Output Format
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
  "source": {
    "page_content": "The dog named Rex lives in London.",
    "metadata": {
      "chunk_id": "1"
    }
  }
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
