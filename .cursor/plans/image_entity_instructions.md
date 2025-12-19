Below is a **complete, future-proof image entity specification**, aligned with:

* scientific PDFs
* multimodal KG construction
* LangChain + Gemini extraction
* Neo4j (or equivalent) graph storage
* later evaluation / ablation studies

I will structure this as:

1. Design principles
2. Image entity schema (core + extended)
3. Relationship types (image-centric)
4. Constraints & validation rules
5. How this supports chunking + extraction
6. What to defer (avoid over-engineering)

---

## 1. Design principles (important to read)

Your image entity must satisfy **four requirements**:

1. **First-class modality**
   Images are not metadata; they are explanatory knowledge units.

2. **Scientifically grounded**
   An image must be linkable to *concepts, methods, equations, and results*.

3. **Chunk-safe & agent-friendly**
   The entity definition must survive 5k-token chunking without losing semantics.

4. **Extensible**
   Future additions (image embeddings, OCR, visual QA) must not break the schema.

---

## 2. Image Entity Schema

### 2.1 Core Image Entity (mandatory)

This is the **minimum required entity**. Every image must satisfy this.

#### `Image` Node

| Field            | Type   | Required | Description                         |
| ---------------- | ------ | -------- | ----------------------------------- |
| `image_id`       | string | ✅        | Stable unique ID (deterministic)    |
| `source_path`    | string | ✅        | Local or relative path to image     |
| `page_number`    | int    | ✅        | PDF page number                     |
| `section_name`   | string | ✅        | Section header containing the image |
| `line_number`    | int    | ✅        | Markdown line number                |
| `caption`        | text   | ✅        | Refined caption text                |
| `document_id`    | string | ✅        | Paper or document identifier        |
| `modality`       | string | ✅        | Always `"image"`                    |

> **Hard rule**:
> If `caption` is missing → the image does NOT enter the KG.

---

### 2.2 Semantic Classification (strongly recommended)

This allows *structured reasoning* over images.

#### Add these attributes to `Image`

| Field                 | Type | Description                                                                                              |
| --------------------- | ---- | -------------------------------------------------------------------------------------------------------- |
| `image_type`          | enum | `architecture`, `diagram`, `plot`, `table_image`, `attention_map`, `equation_render`, `example`, `other` |
| `semantic_role`       | enum | `defines`, `explains`, `illustrates`, `supports_result`, `visualizes_attention`                          |
| `information_density` | enum | `low`, `medium`, `high`                                                                                  |

This enables:

* Filtering noisy plots
* KG evaluation per image type
* Paper-level analysis later

---

### 2.3 Extracted Semantic Payload (critical for KG quality)

These fields are *produced by Gemini*.

| Field               | Type         | Required | Description                |
| ------------------- | ------------ | -------- | -------------------------- |
| `depicted_concepts` | list[string] | ✅        | Concepts explicitly shown  |
| `related_methods`   | list[string] | ⭕        | Methods illustrated        |
| `mentioned_models`  | list[string] | ⭕        | Models/components shown    |
| `visual_keywords`   | list[string] | ⭕        | Salient visual cues        |

> These fields should be populated **before KG insertion**.

---

## 3. Image-Centric Relationship Types

### 3.1 Mandatory relationships

Every image must participate in **at least these**:

```text
(Image)-[:LOCATED_IN]->(Section)
(Image)-[:PART_OF]->(Document)
```

---

### 3.2 Semantic relationships (core)

| Relationship  | From  | To       | Meaning                            |
| ------------- | ----- | -------- | ---------------------------------- |
| `DEPICTS`     | Image | Concept  | Direct visual representation       |
| `ILLUSTRATES` | Image | Method   | Explains how something works       |
| `VISUALIZES`  | Image | Process  | Shows dynamic or flow behavior     |
| `ANNOTATES`   | Image | Equation | Visual explanation of math         |
| `SUPPORTS`    | Image | Result   | Empirical or experimental evidence |

---

### 3.3 Cross-modal grounding

These are extremely important for multimodality:

```text
(Image)-[:GROUNDING_OF]->(TextSpan)
(Image)-[:MENTIONED_NEAR]->(Concept)
```

Use:

* Line number proximity
* Section context

---

## 4. Constraints & Validation Rules

### 4.1 Hard constraints (enforce)

* No image node without:

  * caption
  * page number
  * section
* No orphan image nodes
* At least **1 semantic relationship per image**

---

### 4.2 Soft constraints (warn only)

* Image with >10 concepts → likely over-captioned
* Image with no `image_type` → low quality extraction

---

## 5. How this fits your LangChain + Gemini pipeline

### 5.1 Structured extraction contract

Your agent should output something like:

```json
{
  "image_id": "img_transformer_fig1",
  "image_type": "architecture",
  "semantic_role": "defines",
  "depicted_concepts": [
    "Transformer",
    "Encoder",
    "Decoder",
    "Multi-Head Attention"
  ],
  "relationships": [
    ["DEPICTS", "Transformer"],
    ["ILLUSTRATES", "Multi-Head Attention"]
  ]
}
```

This allows:

* Deterministic KG insertion
* Easy evaluation
* Minimal hallucination risk

---

## 6. What NOT to include

Avoid these for now:

* Pixel-level annotations
* Bounding boxes
* Vision transformers inside KG
* Dense OCR text storage (store separately)

These complicate the system without improving KG quality initially.

---

## 7. Final Image Entity Checklist (printable)

**An image is KG-ready if and only if:**

* ☐ Stable image ID exists
* ☐ Caption is present and refined
* ☐ Page + section + line recorded
* ☐ Image type classified
* ☐ ≥1 concept is depicted
* ☐ ≥1 relationship created

---

## Bottom line

You are not “adding images to text.”
You are **introducing a new modality into a scientific knowledge system**.

This schema ensures:

* Multimodal grounding
* Evaluation-ready graphs
* Zero rework when you scale to other papers