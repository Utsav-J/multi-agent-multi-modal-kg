# Environment Setup & Installation Guide

Follow these steps to set up the Multimodal Knowledge Graph environment.

## 1. Prerequisites

*   **Python 3.10+**
*   **Neo4j Database:** You need a running Neo4j instance (Desktop, Docker, or AuraDB).
*   **Google Gemini API Key:** Required for image captioning and text generation.
*   **uv:** A fast Python package installer and resolver.

## 2. Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd MAS-for-multimodal-knowledge-graph
    ```

2.  **Install Dependencies using `uv`:**
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```
    *Alternatively, if you want to manually install packages:*
    ```bash
    uv add google-genai langchain langchain-community langchain-google-genai langchain-neo4j langchain-text-splitters neo4j pymupdf4llm tiktoken python-dotenv pydantic
    ```

## 3. Configuration

1.  **Create a `.env` file** in the project root:
    ```bash
    touch .env
    ```

2.  **Add your credentials:**
    ```env
    # Google Gemini
    GOOGLE_API_KEY=your_gemini_api_key_here

    # Neo4j Database
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
    ```

## 4. Running the Pipeline

### Option A: Run Everything (Dont do this yet, not tested well)
This command will process up to 5 PDFs from the `data/` folder, generate the graph, and ingest it into Neo4j.
```bash
uv run main_pipeline.py
```

### Option B: Run Agents Individually
You can run specific agents for testing or granular control.

*   **Agent 1 (PDF -> Markdown):**
    ```bash
    uv run agents/1-pdf_processor_agent.py "filename.pdf"
    ```

*   **Agent 2 (Markdown -> Chunks):**
    ```bash
    uv run agents/2-chunker_agent.py "filename_annotated.md"
    ```

*   **Agent 3 (Chunks -> Graph JSONL):**
    ```bash
    uv run agents/3-graph_data_extractor_agent.py "filename_annotated_chunks_5k.jsonl"
    ```

*   **Agent 5 (Ingest Graph Data):**
    ```bash
    uv run agents/5-jsonl_graph_ingestion_agent.py
    ```

## 5. Querying the Graph (not tested)

Once data is ingested, use Agent 6 to ask questions:

```bash
uv run agents/6-graph_query_agent.py "Explain the role of the attention mechanism."
```

