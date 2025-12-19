Below is a **concrete, implementation-oriented to-do list**, ordered exactly in the sequence you should execute it. Each item is phrased as an actionable task with clear completion criteria, so you can track progress and parallelize if needed.

---

# Multimodal Knowledge Graph – Implementation To-Do List

---

## Phase 0 — Preparation & Baseline

### ☐ 0.1 Freeze the current pipeline DONE

* Confirm current flow:

  * PDF → Markdown
  * Markdown → chunking (5k tokens)
  * Chunk → LangChain agent → Gemini → KG
* Ensure this version is reproducible (tag / branch).

**Done when:** You can re-run the text-only KG extraction end-to-end.

---

### ☐ 0.2 Define image entity requirements DONE

* Decide minimum attributes for an `Image` entity:

  * `id`
  * `source_path`
  * `page_number`
  * `figure_label` (if available)
  * `caption`
* Decide initial relationship types:

  * `DEPICTS`
  * `ILLUSTRATES`
  * `LOCATED_IN`
  * `REFERENCES`

**Done when:** You have a written mini-schema (even a markdown note).

---

## Phase 1 — Image Extraction & Indexing

### ☐ 1.1 Detect all image references in Markdown DONE

* Parse Markdown to find:

  ```markdown
  ![](path/to/image.png)
  ```
* Collect:

  * Image path
  * Position in document
  * Surrounding section header (if any)

**Done when:** You can print a list of all images with page and section info.

---

### ☐ 1.2 Assign stable image IDs DONE

* Generate deterministic IDs:

  * `img_<paper_id>_<page>_<index>`
* Maintain an image manifest:

  ```json
  {
    "img_fig1": {
      "path": "...png",
      "page": 2,
      "section": "Model Architecture"
    }
  }
  ```

**Done when:** Every image has a stable, reproducible ID.

---

## Phase 2 — Image Captioning (Local, Cheap)

### ☐ 2.1 Integrate BLIP-2 captioning DONE

* Load:

  * `Salesforce/blip2-flan-t5-base`
* Build a small script:

  * Input: image path
  * Output: raw caption text

**Done when:** You can caption a single image locally.

---

### ☐ 2.2 Batch caption all images DONE

* Run BLIP-2 on all extracted images
* Store:

  * Raw captions
  * Timestamp / model version

**Done when:** Every image has an associated raw caption.

---

### ☐ 2.3 (Optional but recommended) Caption refinement with Gemini DONE

* Prompt Gemini with:

  * Raw caption
  * Surrounding Markdown text (±1 section)
* Ask for:

  * Scientifically precise description
  * Mention of depicted concepts

**Done when:** Captions read like figure descriptions from a paper.

---

## Phase 3 — Markdown Rewriting (Critical Step)

### ☐ 3.1 Replace inline images with structured image blocks DONE

* Replace:

  ```markdown
  ![](image.png)
  ```
* With:

  ```markdown
  :::image
  id: img_fig1
  source: image.png
  page: 2
  figure_label: Figure 1
  caption: >
    Refined caption text...
  :::
  ```

**Done when:** No raw `![]()` image syntax remains.

---

### ☐ 3.2 Preserve document structure DONE

* Ensure image blocks appear:

  * Immediately after figure mentions
  * Inside correct section headers

**Done when:** Reading the Markdown still feels natural.

---

### ☐ 3.3 Validate chunk safety DONE

* Confirm image blocks do not exceed chunk limits
* Ensure each block is self-contained

**Done when:** Chunking does not split image blocks.

---

## Phase 4 — Chunking & Ingestion

### ☐ 4.1 Update chunker to treat image blocks atomically DONE

* Prevent splitting inside `:::image` blocks
* Chunk size ≈ 5000 tokens

**Done when:** Each chunk contains whole image blocks.

---

### ☐ 4.2 Tag chunks with multimodal metadata

* Add chunk metadata:

  * `contains_image: true/false`
  * `image_ids: [...]`

**Done when:** Downstream agents can detect image presence.

---

## Phase 5 — Entity & Relationship Extraction (LangChain + Gemini)

### ☐ 5.1 Update extraction prompt

Explicitly instruct Gemini:

* Images are first-class entities
* Extract:

  * Image entities
  * Concepts depicted
  * Relationships between image ↔ text entities

**Done when:** Prompt explicitly mentions images as entities.

---

### ☐ 5.2 Map extracted data to KG schema

* Image → `Image` node
* Caption concepts → `Concept` nodes
* Relationships:

  * `Image → DEPICTS → Concept`
  * `Image → LOCATED_IN → Section`

**Done when:** Images appear as nodes in the KG.

---

### ☐ 5.3 Merge image and text entities

* Deduplicate concepts:

  * “Multi-Head Attention” in text == image caption
* Use string + embedding similarity if needed

**Done when:** Images link to existing concept nodes.

---

## Phase 6 — Graph Validation & Quality Control

### ☐ 6.1 Sanity-check image grounding

* For each image:

  * At least 1 `DEPICTS` relation
  * Correct section linkage

**Done when:** No orphan image nodes exist.

---

### ☐ 6.2 Visual KG inspection (Neo4j)

* Query:

  ```cypher
  MATCH (i:Image)-[r]->(c) RETURN i, r, c
  ```
* Confirm relationships are meaningful

**Done when:** Image subgraph looks semantically correct.

---

## Phase 7 — Optional Enhancements (High ROI)

### ☐ 7.1 Add CLIP embeddings for images

* Store vector per image
* Enable cross-modal similarity later

---

### ☐ 7.2 Add “explains” vs “illustrates” relation typing

* Distinguish:

  * Architectural diagrams
  * Experimental plots
  * Attention visualizations

---

### ☐ 7.3 Add provenance metadata

* Link images to:

  * Page
  * Figure number
  * PDF source

---

## Final Outcome Checklist

By the end, you should have:

* Images as **explicit KG entities**
* Images grounded in **scientific concepts**
* Multimodal chunks processed by LangChain + Gemini
* A KG that supports **text ↔ image reasoning**

If you want, next we can:

* Convert this into a **GitHub issue checklist**
* Design the **Neo4j constraints & indexes**
* Write the **exact Gemini extraction prompt** you should use
