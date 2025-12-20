# Multimodal Knowledge Graph Pipeline

A comprehensive Multi-Agent System (MAS) for processing research papers into a Neo4j Knowledge Graph, capable of handling text, images, and cross-document entity relationships.

## 🚀 Key Features

### 1. Intelligent PDF Processing (Agent 1)
*   **Automated Conversion:** Converts PDF research papers into clean Markdown using `pymupdf4llm`.
*   **Smart Image Annotation:**
    *   Extracts images from PDFs.
    *   Automatically generates AI captions for images using **Google Gemini 1.5 Flash**.
    *   *Optimization:* Only triggers annotation if a document has ≤ 5 images to save costs and time.
*   **Batch Processing:** Can process single files or scan entire directories.

### 2. Dual-Granularity Chunking (Agent 2)
*   **Token-Based Splitting:** Uses `tiktoken` for precise context window management.
*   **Two Outputs per Document:**
    *   **5k Token Chunks:** Optimized for Knowledge Graph construction (capturing broad context and relationships).
    *   **2k Token Chunks:** Optimized for standard RAG (Retrieval Augmented Generation) and search.

### 3. Context-Aware Knowledge Extraction (Agent 3)
*   **Graph Construction:** Extracts Entities (Nodes) and Relationships (Edges) from text chunks using LLMs.
*   **Global Entity Registry:** Maintains a `global_entity_registry.json` to ensure consistency across documents.
    *   *Example:* If "Attention Mechanism" is identified in Paper A, Paper B will reuse this exact entity ID instead of creating "Attention".
*   **Redundancy Handling:** Prevents duplicate node creation for the same concepts across the corpus.

### 4. Batch Graph Ingestion (Agent 5)
*   **Neo4j Integration:** Connects to a Neo4j vector database.
*   **Bulk Loading:** Automatically finds all generated Graph JSONL files and ingests them into the database in one go.

### 5. Natural Language Querying (Agent 6)
*   **Text-to-Cypher:** Converts user questions (e.g., "How does the attention mechanism work?") into Cypher queries.
*   **Graph QA:** Executes the query against Neo4j and synthesizes a natural language answer based on the retrieved subgraph.
*   **Detailed behavior**: See `agents/QUERY_AGENT_BEHAVIOR.md` (keep this updated whenever Agent 6 changes).

### 6. Pipeline Orchestrator
*   **End-to-End Automation:** `main_pipeline.py` automates the entire flow from raw PDF to queryable database.
*   **Dynamic Flow:** Automatically handles file dependencies (e.g., using raw vs. annotated markdown based on Agent 1's decision).

## 📂 Project Structure

```
.
├── agents/
│   ├── 1-pdf_processor_agent.py       # PDF -> Markdown + Image Captioning
│   ├── 2-chunker_agent.py             # Markdown -> 5k/2k Token Chunks
│   ├── 3-graph_data_extractor_agent.py # Chunks -> Graph Data (JSONL)
│   ├── 5-jsonl_graph_ingestion_agent.py # JSONL -> Neo4j Database
│   └── 6-query_agent.py               # QA Interface (see agents/QUERY_AGENT_BEHAVIOR.md)
├── data/                              # Input PDFs
├── knowledge_graph_outputs/           # Generated Graph Data & Registry
├── markdown_outputs/                  # Intermediate Markdown & Images
├── chunking_outputs/                  # JSONL Chunks
└── main_pipeline.py                   # Master Orchestrator Script
```

