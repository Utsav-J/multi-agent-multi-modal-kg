# Image Extraction, Captioning, and Processing Pipeline

This document describes the multimodal image processing pipeline for scientific documents. The pipeline extracts images from markdown files, generates structured captions using vision-language models, and enriches markdown documents with semantic image metadata suitable for knowledge graph construction.

## Overview

The pipeline consists of three main stages:

1. **Image Extraction and Metadata Embedding** - Identifies images in markdown files and creates stable image identifiers
2. **Image Captioning** - Generates structured, scientific captions using Gemini vision-language model
3. **Markdown Enrichment** - Merges caption data back into markdown files as embedded JSON metadata

The end result is markdown files containing rich, structured image metadata that can be used for downstream knowledge graph extraction.

---

## Stage 1: Image Extraction and Metadata Embedding

**Module**: `image_extraction_utils.py`  
**Function**: `assign_stable_image_ids()`

### Purpose

Extracts image references from markdown files and assigns deterministic, stable image identifiers. Embedds image metadata as JSON snippets directly within the markdown file structure.

### Process Flow

1. **Parse Markdown File**
   - Scan markdown content for image syntax: `![](path/to/image.png)`
   - Extract image paths from markdown image syntax

2. **Extract Image Metadata**
   - **Page Number**: Extracted from filename pattern `<paper_id>.pdf-<page>-<index>.png`
   - **Section Header**: Identified by scanning backwards for section headers (patterns: `## Title`, `**N** **Title**`, etc.)
   - **Line Position**: Track line number for reference

3. **Generate Stable Image IDs**
   - Format: `img_<paper_id>_<page>_<index>`
   - Example: `img_attention_is_all_you_need_2_0`
   - IDs are deterministic based on filename pattern

4. **Embed Metadata as JSON**
   - Replace original image markdown with JSON metadata blocks
   - Format:
     ```json
     "img_paper_id_2_0": {
       "path": "E:/.../attention_is_all_you_need.pdf-2-0.png",
       "page": 2,
       "section": "Model Architecture"
     }
     ```

### Output

A new markdown file (suffix `_with_image_ids.md`) containing:
- Original markdown content (images removed)
- Embedded JSON blocks with image metadata
- Stable image IDs for reference

---

## Stage 2: Image Captioning

**Module**: `image_captioning_utils.py`  
**Function**: `generate(image_path: str)`

### Purpose

Generates structured, information-dense captions for scientific images using Google's Gemini 2.5 Flash vision-language model with constrained JSON output.

### Process Flow

1. **Image Input**
   - Accepts image file path (PNG, JPG, etc.)
   - Reads image bytes for API transmission

2. **Vision-Language Model Processing**
   - **Model**: Gemini 2.5 Flash (multimodal)
   - **Method**: Structured output with JSON schema validation
   - **System Prompt**: Scientific document analysis focused on knowledge graph construction

3. **Structured Output Schema**

   The model returns a JSON object with the following fields:

   - `image_relevance`: `"high"` | `"low"` - Scientific relevance assessment
   - `image_type`: Enum (architecture, diagram, plot, attention_map, table_image, equation_render, logo, other)
   - `semantic_role`: Enum (defines, explains, illustrates, supports_result, decorative)
   - `caption`: String - Information-dense caption (1-3 sentences)
   - `depicted_concepts`: List[str] - Key scientific concepts visible in the image
   - `confidence`: `"high"` | `"medium"` | `"low"` - Confidence in the caption

4. **Caption Quality Guidelines**

   The system prompt enforces:
   - Precision over completeness
   - Scientific terminology consistency
   - Only describe what is visually present (no inference)
   - Avoid narrative phrasing ("the paper shows...")
   - Focus on entities and relationships suitable for knowledge graphs

### Output

JSON string containing structured caption data suitable for merging with existing image metadata.

### Example Output

```json
{
  "image_relevance": "high",
  "image_type": "architecture",
  "semantic_role": "defines",
  "caption": "The image illustrates the architecture of a Transformer model, consisting of an Encoder and a Decoder...",
  "depicted_concepts": [
    "Transformer architecture",
    "Encoder",
    "Decoder",
    "Multi-Head Attention"
  ],
  "confidence": "high"
}
```

---

## Stage 3: Markdown Enrichment with Captions

**Module**: `markdown_image_processing.py`  
**Function**: `caption_markdown_images(markdown_path: str)`

### Purpose

Processes markdown files containing embedded image metadata JSON blocks, enriches each image with generated captions, and produces a final enriched markdown file.

### Process Flow

1. **Parse Markdown File**
   - Read markdown content line by line
   - Identify fenced JSON code blocks (```json ... ```)
   - Detect image metadata snippets using pattern: `"img_<id>": { ... }`

2. **Extract Image Metadata**
   - For each image metadata block:
     - Parse image ID
     - Extract existing metadata (path, page, section)
     - Validate image path exists

3. **Generate Captions** (per image)
   - Call `generate(image_path)` from `image_captioning_utils.py`
   - Handle errors gracefully (fallback to original metadata on failure)
   - Log processing statistics

4. **Merge Caption Data**
   - Combine existing metadata with caption output
   - Preserve original metadata fields (path, page, section)
   - Add caption fields (image_relevance, image_type, semantic_role, caption, depicted_concepts, confidence)
   - Order fields for readability

5. **Write Enriched Markdown**
   - Format merged metadata as JSON snippet
   - Preserve original markdown structure
   - Create new file with suffix `_with_captions.md`

### Output Format

The enriched JSON blocks contain both original metadata and caption data:

```json
"img_attention_is_all_you_need_2_0": {
  "path": "E:/.../attention_is_all_you_need.pdf-2-0.png",
  "page": 2,
  "section": "Model Architecture",
  "image_relevance": "high",
  "image_type": "architecture",
  "semantic_role": "defines",
  "caption": "The image illustrates the architecture of a Transformer model...",
  "depicted_concepts": [
    "Transformer architecture",
    "Encoder",
    "Decoder"
  ],
  "confidence": "high"
}
```

---

## Complete Pipeline Workflow

### Sequential Processing

```
[Original Markdown]
    ↓
[Stage 1: Image Extraction]
Extract images → Generate IDs → Embed metadata JSON
    ↓
[Markdown with Image IDs]
    ↓
[Stage 2: Image Captioning] (per image)
Read image → Gemini API → Structured caption JSON
    ↓
[Stage 3: Markdown Enrichment]
Parse JSON blocks → Merge captions → Write enriched markdown
    ↓
[Enriched Markdown with Captions]
```

### Data Flow

1. **Input**: Raw markdown file with image references
2. **Intermediate 1**: Markdown with embedded image metadata (paths, pages, sections)
3. **Intermediate 2**: Structured caption JSON for each image
4. **Output**: Enriched markdown with complete image metadata including captions

---

## Key Design Decisions

### Stable Image IDs

- **Format**: `img_<paper_id>_<page>_<index>`
- **Rationale**: Deterministic IDs enable consistent reference across pipeline stages
- **Benefits**: 
  - Enables tracking images through processing
  - Supports deduplication
  - Facilitates knowledge graph entity linking

### Embedded JSON Metadata

- **Approach**: Store metadata directly in markdown as fenced JSON blocks
- **Rationale**: 
  - Single-file format (no separate manifest files)
  - Human-readable and editable
  - Preserves document structure
- **Trade-off**: Slightly increases markdown file size

### Structured Captioning

- **Schema**: Fixed JSON schema with enums and typed fields
- **Rationale**:
  - Enables programmatic processing
  - Ensures consistent data structure
  - Supports validation and error handling
- **Quality Control**: System prompts enforce scientific precision over completeness

### Error Handling

- **Graceful Degradation**: Failed captioning preserves original metadata
- **Logging**: Comprehensive logging for debugging and monitoring
- **Validation**: Path existence checks before captioning
- **Fallback**: Original metadata retained on any processing error

---

## Integration with Knowledge Graph Pipeline

The enriched markdown files serve as input for downstream knowledge graph extraction:

1. **Chunking**: Markdown is chunked (including embedded image metadata)
2. **Graph Extraction**: 
   - Text chunks processed for entities/relationships
   - Image metadata processed separately (see `agents/3-graph_data_extractor_agent.py`)
3. **Graph Construction**:
   - Image entities created from metadata
   - Caption text used for concept extraction
   - Depicted concepts linked to image entities
   - Images linked to documents and sections

### Metadata Utilization

- **`caption`**: Extracted for entities and relationships
- **`depicted_concepts`**: Directly converted to Concept nodes
- **`semantic_role`**: Used to classify image-document relationships
- **`image_type`**: Stored as entity property
- **`path`**: Used for provenance tracking

---

## Technical Specifications

### Dependencies

- **Google Gemini API**: Vision-language model for captioning
- **Python Standard Library**: File I/O, JSON processing, regex
- **Pathlib**: Cross-platform path handling

### Performance Considerations

- **Sequential Processing**: Images processed one-by-one (API rate limits)
- **Error Isolation**: Individual image failures don't block pipeline
- **Logging**: Separate log files per processing run

### File Naming Conventions

- Input: `<paper>_raw.md`
- Stage 1 output: `<paper>_raw_with_image_ids.md`
- Stage 2 output: `<paper>_raw_with_image_ids_with_captions.md`
- (Further stages may add additional suffixes)

---

## Usage Examples

### Stage 1: Extract Images and Assign IDs

```bash
uv run markdown_outputs/image_extraction_utils.py \
  attention_is_all_you_need_raw.md \
  --assign-ids \
  --output-markdown attention_is_all_you_need_raw_with_image_ids.md
```

### Stage 2: Generate Caption for Single Image

```bash
uv run markdown_outputs/image_captioning_utils.py \
  --image "E:/path/to/image.png"
```

### Stage 3: Process All Images in Markdown

```bash
uv run markdown_outputs/markdown_image_processing.py \
  attention_is_all_you_need_raw_with_image_ids.md
```

---

## Future Enhancements

Potential improvements to the pipeline:

1. **Batch Processing**: Process multiple images in parallel (with rate limiting)
2. **Caption Caching**: Cache captions to avoid re-processing
3. **Quality Metrics**: Track caption quality scores across documents
4. **Alternative Models**: Support for other vision-language models
5. **Incremental Updates**: Re-process only changed or new images

---

## References

- **Image Extraction**: `markdown_outputs/image_extraction_utils.py`
- **Image Captioning**: `markdown_outputs/image_captioning_utils.py`
- **Markdown Processing**: `markdown_outputs/markdown_image_processing.py`
- **Graph Extraction Integration**: `agents/3-graph_data_extractor_agent.py`


