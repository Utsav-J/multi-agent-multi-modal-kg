# Pending Implementation Tasks

This document tracks features and improvements outlined in the original roadmap that are yet to be fully realized or optimized.

## 🚧 Future Improvements

### 1. Advanced RAG Integration (Agent 4)
*   **Status:** *Skipped in current iteration.*
*   **Goal:** The original plan implicitly separated "Graph Construction" from "Standard RAG". While we generate 2k token chunks for RAG, we currently rely purely on the Graph Agent (Agent 6) for Q&A.
*   **Todo:** Implement a hybrid retriever that combines:
    *   Vector Search (using the 2k chunks).
    *   Graph Search (Cypher queries).
    *   Re-ranking for final answer generation.

### 2. Dynamic Schema Evolution
*   **Status:** *Basic Implementation.*
*   **Goal:** Currently, the `Global Entity Registry` tracks entity *IDs* to prevent duplicates.
*   **Todo:** Expand the registry to track *Relationships* and *Node Types* more strictly to prevent schema drift (e.g., ensuring "Person" nodes don't accidentally get labeled as "Human" in a different paper).

### 4. Robust Error Recovery in Pipeline
*   **Status:** *Basic.*
*   **Goal:** The pipeline runs sequentially.
*   **Todo:** Implement checkpointing. If Agent 3 fails on Document 4 of 5, the pipeline should be able to resume from that exact point without re-processing the first 3 documents.

### 5. Multi-Modal Querying
*   **Status:** *Partial.*
*   **Goal:** We extract and caption images, but the Query Agent primarily searches text/graph nodes.
*   **Todo:** Enable the Query Agent to return relevant images (from `markdown_outputs/images`) as part of the answer context.

