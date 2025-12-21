## `agents/1-pdf_processor_agent.py` — PDF Processor Agent

### Purpose (what this agent is for)

This agent is the **ingestion entrypoint** for the pipeline. It converts one (or many) PDFs from `data/` into Markdown under `markdown_outputs/`, and optionally annotates extracted images with AI-generated captions.

It wraps two concrete tools (pure functions + side effects) in a one-shot LangChain agent:

- **`convert_pdf_to_markdown_tool(pdf_filename)`**: PDF → raw Markdown + extracted images
- **`annotate_markdown_tool(markdown_filename)`**: raw Markdown → annotated Markdown (image captioning)

### How to run

From repo root:

- Process one PDF:
  - `uv run agents/1-pdf_processor_agent.py attention_is_all_you_need.pdf`
- Process all PDFs in `data/`:
  - `uv run agents/1-pdf_processor_agent.py`
- Force skip annotation:
  - `uv run agents/1-pdf_processor_agent.py attention_is_all_you_need.pdf --no-annotate`

### Where it sits in the pipeline

Downstream stages assume this agent has produced:

- A Markdown file in `markdown_outputs/` (raw or annotated)
- Extracted images in `markdown_outputs/images/`
- (Optionally) structured image metadata embedded into the Markdown (used later by the KG image-entity extractor)

### Inputs

- **CLI positional arg**: `filename` (optional)
  - If provided: processes that single PDF (from `data/<filename>`)
  - If omitted: scans `data/` for `*.pdf` and processes all found PDFs
- **CLI flag**: `--no-annotate`
  - Forces the agent to skip annotation regardless of image count

### Outputs / artifacts

For each processed PDF `data/<base>.pdf`:

- **Raw Markdown**: `markdown_outputs/<base>_raw.md`
- **Extracted images directory**: `markdown_outputs/images/` (images are written here by the PDF conversion utility)
- **Annotated Markdown (optional)**: `markdown_outputs/<base>_annotated.md`

Return value from `main()`:

- Returns a Python list of generated filenames (best-effort inference based on the agent’s last message).

### Environment / dependencies

- Loads environment variables via `dotenv.load_dotenv()`.
- Uses `langchain_google_genai.ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)`.
- Relies on project utilities:
  - `utils.process_pdf.convert_pdf_to_markdown(input_pdf, intermediate_md, images_dir)`
  - `utils.process_pdf.annotate_markdown_images(input_md, final_md)`

### Tool behavior (implementation details)

#### 1) `convert_pdf_to_markdown_tool(pdf_filename: str) -> str`

- **Input location**: `data/<pdf_filename>`
- **Writes**:
  - Markdown to `markdown_outputs/<base>_raw.md`
  - Images to `markdown_outputs/images/`
- **Returns**: a human-readable status string including image count
- **Failure modes**:
  - Missing input file → returns `"Error: Input file not found at ..."`
  - Any exception → returns `"Error during PDF conversion: ..."`

#### 2) `annotate_markdown_tool(markdown_filename: str) -> str`

- **Input location**: `markdown_outputs/<markdown_filename>`
- **Writes**:
  - Annotated markdown to `markdown_outputs/<base>_annotated.md` where `<base>` is derived by removing `_raw` from the input stem.
- **Important side effect**: The annotation function writes output to disk; the tool does not return the markdown content itself.

### Agent decision rule (annotation gating)

The agent uses a system prompt that enforces:

- Always convert the PDF first.
- Only annotate images if the conversion reports **≤ 5 images**.
- If **> 5 images**, skip annotation.
- If `--no-annotate` is provided, the user message explicitly instructs skipping annotation regardless of count.

### Special features / noteworthy behaviors

- **Batch mode**: if no `filename` is provided, it processes all PDFs in `data/`.
- **Heuristic output selection**: after agent execution, it inspects the agent’s final message text to decide whether the output was annotated. This is a *string-match heuristic* and can be wrong if the message wording changes.
- **Path discipline**: all I/O uses `Path(__file__).resolve().parent.parent` as the project root, so it’s robust to being run from different working directories.

### Reproducibility notes (for research writing)

- Captioning is an LLM-based step; reproducibility depends on:
  - model version (`gemini-2.5-flash`)
  - prompt + deterministic settings (`temperature=0`)
  - upstream PDF extraction determinism (PDF parsing/image extraction may vary by library versions)
- The gating threshold (≤ 5 images) is a **design choice** that trades cost/time vs. richer multimodal grounding.

### Paper-ready “Method” description (suggested wording)

We convert each source PDF into Markdown while extracting embedded figures. When the number of extracted figures is small (≤5), we invoke an image-captioning stage to attach structured captions and figure metadata to the Markdown. The resulting Markdown serves as the canonical text+figure representation for downstream chunking, retrieval indexing, and knowledge graph construction.


