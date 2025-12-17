---
name: Create Hybrid RAG Query Agent (No Vector Store Init)
overview: Implement `agents/6-query_agent.py` to query an existing vector store (FAISS) and Neo4j graph, assuming vector store creation is handled elsewhere.
todos:
  - id: create-agent-file
    content: Create `agents/6-query_agent.py` with vector store loading logic.
    status: pending
  - id: implement-rag-tool
    content: Implement `rag_retrieval_tool` (retrieval only).
    status: pending
    dependencies:
      - create-agent-file
  - id: implement-graph-tool
    content: Implement `graph_retrieval_tool` with GraphCypherQAChain.
    status: pending
    dependencies:
      - create-agent-file
  - id: implement-main-loop
    content: Implement main agent loop and synthesis logic.
    status: pending
    dependencies:
      - implement-rag-tool
      - implement-graph-tool
---

# Create Hybrid RAG Query Agent

This plan involves creating `agents/6-query_agent.py` which will query an *existing* vector store and a graph database (Neo4j). The vector store initialization/creation is explicitly out of scope for this agent (it will assume the store exists on disk or connect to a service).

## 1. Prerequisites & Setup

- **Vector Store**: The agent assumes a FAISS index (or similar) already exists at a specified path (e.g., `vector_store/`).
- **Graph Database**: Neo4j connection parameters must be available via environment variables.

## 2. Implementation: `agents/6-query_agent.py`

### Key Components:

1.  **Vector Store Loading**:

    - Function `load_vector_store()`:
        - Loads the FAISS index from disk (`vector_store/` or configured path).
        - Uses `GoogleGenerativeAIEmbeddings`.

2.  **RAG Tool (`rag_tool`)**:

    - Wraps retrieval from the loaded vector store.
    - **Logic**:
        - Input: Natural language query.
        - Process: 
            - Generate sub-queries (using LLM).
            - Retrieve documents for each sub-query.
            - Reciprocal Rank Fusion (optional) or simple deduping.
        - Output: List of relevant text chunks.

3.  **Graph Tool (`graph_tool`)**:

    - Uses `GraphCypherQAChain`.
    - **Logic**:
        - Input: Natural language query.
        - Process: LLM translates to Cypher -> Execute on Neo4j -> Return results.
        - Output: Graph data/answer.

4.  **Main Agent**:

    - Uses `create_tool_calling_agent` or `create_agent` with `ChatGoogleGenerativeAI`.
    - System Prompt: Instructions to use both tools to gather information and then synthesize a complete answer.
    - Logging: Use `logging` module to track retrieval and query execution.

### Logic Flow:

1.  User inputs query (e.g., "what is attention?").
2.  Agent calls `rag_tool` (which internally rephrases and retrieves).
3.  Agent calls `graph_tool` (which converts to Cypher and queries).
4.  Agent combines outputs and generates final response.

## 3. Execution

- The script will have a `main()` block.
- It will log the Cypher query and retrieved chunks.

## 4. Dependencies

- [agents/5-jsonl_graph_ingestion_agent.py](agents/5-jsonl_graph_ingestion_agent.py) (Reference for Neo4j)