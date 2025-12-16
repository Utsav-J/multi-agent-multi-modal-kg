---
name: Multimodal KG Pipeline Roadmap
overview: I will implement a complete pipeline that processes PDFs into a Neo4j Knowledge Graph, handling image annotation, dual-chunking, cross-document entity redundancy, and natural language querying.
todos:
  - id: agent-1-update
    content: Refactor Agent 1 for auto-annotation logic and batch support
    status: completed
  - id: agent-2-update
    content: Refactor Agent 2 for dual token-based chunking (5k & 2k)
    status: completed
  - id: agent-3-update
    content: Refactor Agent 3 to implement Global Entity Registry for redundancy handling
    status: completed
    dependencies:
      - agent-2-update
  - id: create-agent-6
    content: Create Agent 6 for Graph Querying (Text -> Cypher -> Answer)
    status: completed
    dependencies:
      - agent-3-update
  - id: create-orchestrator
    content: Create main_pipeline.py to orchestrate the entire flow
    status: completed
    dependencies:
      - agent-1-update
      - agent-2-update
      - agent-3-update
      - create-agent-6
---

# Multimodal Knowledge Graph Pipeline Roadmap

This plan outlines the steps to upgrade the existing agentic workflow to meet the requirements for processing research papers, building a knowledge graph, and enabling Q&A.

## 1. Upgrade PDF Processor Agent (Agent 1)

**File:** [`agents/1-pdf_processor_agent.py`](agents/1-pdf_processor_agent.py)

- **Goal:** Intelligent image annotation.
- **Changes:**
  - Modify `convert_pdf_to_markdown_tool` to count extracted images in the output directory.
  - Implement logic: IF image_count <= 5 THEN trigger `annotate_markdown_tool` ELSE skip.
  - Support batch processing of multiple PDFs from the `data/` directory.

## 2. Upgrade Chunker Agent (Agent 2)

**File:** [`agents/2-chunker_agent.py`](agents/2-chunker_agent.py)

- **Goal:** Dual-granularity chunking.
- **Changes:**
  - Switch to token-based splitting (using `tiktoken`) to ensure accurate "5000 token" and "2000 token" sizes.
  - Output two JSONL files per document:
    - `{doc}_chunks_5k.jsonl` (for Graph Construction)
    - `{doc}_chunks_2k.jsonl` (for RAG/Search)

## 3. Upgrade Graph Data Extractor (Agent 3)

**File:** [`agents/3-graph_data_extractor_agent.py`](agents/3-graph_data_extractor_agent.py)

- **Goal:** Context-aware entity extraction across documents.
- **Changes:**
  - Implement a `GlobalEntityRegistry` (saved as JSON) to persist seen entities across different documents.
  - Load this registry before processing each document so the LLM knows which entities already exist ("attention mechanism" -> "Attention").
  - Target the `_5k.jsonl` chunk files for extraction.

## 4. Graph Ingestion Agent (Agent 5)

**File:** [`agents/5-jsonl_graph_ingestion_agent.py`](agents/5-jsonl_graph_ingestion_agent.py)

- **Goal:** Batch ingestion.
- **Changes:**
  - Ensure it iterates through all generated `*_graph.jsonl` files and uploads them to Neo4j.

## 5. Create Graph Query Agent (New Agent 6)

**File:** `agents/6-graph_query_agent.py`

- **Goal:** Question Answering.
- **Implementation:**
  - Create a new agent using LangChain's `GraphCypherQAChain` (or similar).
  - It will accept a natural language query ("tell me about attention"), generate Cypher, execute it against Neo4j, and synthesize an answer.

## 6. Create Pipeline Orchestrator

**File:** `main_pipeline.py`

- **Goal:** End-to-end automation.
- **Implementation:**
  - A master script that sequentially calls Agents 1 -> 2 -> 3 -> 5 for all PDF files in the `data/` folder.
  - Handles the flow of filenames and status updates.

## Diagram

```mermaid
flowchart TD
    PDFs[PDF Documents] --> Agent1[Agent 1: PDF Processor]
    Agent1 -- "Extracted Images" --> ImgCheck{<= 5 Images?}
    ImgCheck -- Yes --> Annotate[Annotate Images]
    ImgCheck -- No --> Skip[Skip Annotation]
    Annotate --> MD[Markdown]
    Skip --> MD
    MD --> Agent2[Agent 2: Chunker]
    Agent2 --> Chunk5k[5k Token Chunks JSONL]
    Agent2 --> Chunk2k[2k Token Chunks JSONL]
    Chunk5k --> Agent3[Agent 3: Graph Extractor]
    Registry[(Entity Registry)] <--> Agent3
    Agent3 --> GraphJSONL[Graph Data JSONL]
    GraphJSONL --> Agent5[Agent 5: Ingestion]
    Agent5 --> Neo4j[(Neo4j DB)]
    User((User)) --> Agent6[Agent 6: Query Agent]
    Agent6 <--> Neo4j
```