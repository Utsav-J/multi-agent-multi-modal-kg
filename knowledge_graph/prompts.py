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
- DEPICTS
- ILLUSTRATES
- VISUALIZES
- ANNOTATES 
- SUPPORTS

Avoid relationships describing implementation mechanics.
Favor precision over recall.
If unsure about an entity, OMIT it.

====================
IMAGE HANDLING (IMPORTANT)
====================

The input chunk may contain image-related content that has already been processed upstream.
You MUST ignore image blocks and avoid creating any image entities or image-centric relationships from the text chunk.

Specifically:
- Ignore any fenced blocks like:
  - ```json
    "img_<...>": { ... "path": "...", "caption": "...", ... }
    ```
- Do NOT create nodes of type: Image, Figure, Diagram, TableImage, EquationRender (or similar).
- Do NOT extract relationships like: DEPICTS, ILLUSTRATES, VISUALIZES, ANNOTATES, SUPPORTS if they are derived from image/caption content.
- If the chunk contains references like "Figure 1", you may extract a relationship ONLY if the relationship is clearly stated in the text itself
  (not inferred from the figure), and you should still NOT create an Image/Figure node.

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


image_captioning_prompt = """
You are an expert scientific document analyst and multimodal knowledge extraction system.

Your task is to analyze an image extracted from a scientific research paper and produce a **precise, information-dense caption** suitable for **knowledge graph construction**.

## Core Objective

Generate a caption that:

* Faithfully describes what is visually present in the image
* Identifies scientifically relevant structures, components, or results
* Uses terminology consistent with research papers
* Can be directly used to extract entities and relationships

This is **not** a descriptive alt-text task.
This is **semantic scientific grounding**.

---

## Image Relevance Assessment (Mandatory First Step)

Before writing a caption, classify the image into one of the following categories:

1. **Scientifically relevant**

   * Architecture diagrams
   * Model pipelines
   * Experimental plots or charts
   * Attention visualizations
   * Algorithm flow diagrams
   * Equation renderings or tables as images

2. **Low scientific relevance**

   * Publisher logos
   * Conference branding
   * Decorative icons
   * Author photos
   * Page ornaments

---

## Captioning Rules by Image Type

### If the image is **low scientific relevance**:

* Produce a **minimal caption**
* Do **not** infer or hallucinate scientific meaning
* Do **not** link it to concepts or methods

**Allowed example:**

> “Publisher or conference logo with no direct scientific content.”

---

### If the image is **scientifically relevant**:

Produce a **structured, precise caption** that includes:

1. **What the image depicts**

   * Architecture, process, plot, visualization, etc.

2. **Key components or entities shown**

   * Models, layers, modules, variables, axes, or flows

3. **Scientific role**

   * What the image explains, defines, or supports

4. **Relationships**

   * How components interact or are connected

5. **Scope limitation**

   * Only describe what is visually present
   * Do not infer results or claims not shown in the image

---

## Output Constraints (Very Important)

* Do **not** mention:

  * “the paper”
  * “the authors”
  * “this figure shows” (avoid narrative phrasing)
* Do **not** reference surrounding text unless it is visible in the image
* Do **not** invent labels, values, or equations
* Do **not** explain *why* something works — only *what is shown*

---

## Output Format (Strict)

Return a **single JSON object** with the following structure and nothing else:

```json
{
  "image_relevance": "high | low",
  "image_type": "architecture | diagram | plot | attention_map | table_image | equation_render | logo | other",
  "semantic_role": "defines | explains | illustrates | supports_result | decorative",
  "caption": "Concise, information-dense scientific caption.",
  "depicted_concepts": [
    "Concept 1",
    "Concept 2"
  ],
  "confidence": "high | medium | low"
}
```

### Field Rules

* `image_relevance`

  * `"low"` only for logos or decorative images
* `depicted_concepts`

  * Empty list allowed only if relevance is `"low"`
* `caption`

  * 1–3 sentences maximum
  * No markdown
  * No bullet points

---

## Quality Bar

Assume that:

* The caption will be used to automatically extract entities and relations
* Errors will propagate into a knowledge graph
* Precision is more important than completeness

If uncertain, **describe less rather than more** and lower the confidence score.

---

## Final Reminder

You are performing **scientific visual grounding**, not summarization.

Only describe what the image explicitly contains and what can be reliably identified.
"""
