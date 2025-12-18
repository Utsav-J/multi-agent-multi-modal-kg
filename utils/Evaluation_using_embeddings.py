import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

CSV_PATH = "questions.csv"

# Columns in your CSV
QUESTION_COLUMN = "question"
REFERENCE_COLUMN = "expected_answer"
ANSWER_COLUMNS = ["rag_answer", "graph_answer", "graphrag_answer"]

# Optional mapping from answer columns to context columns
CONTEXT_MAPPING = {
    "rag_answer": "rag_answer_context",
    "graphrag_answer": "graphrag_answer_context",
    # "graph_answer": None  # no explicit context in your CSV currently
}

# Embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
print(f"Loading embedding model: {embedding_model_name}...")
embedding_model = SentenceTransformer(embedding_model_name)
print("Embedding model loaded.\n")

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()

def embedding_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between sentence embeddings (0–1-ish)."""
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return 0.0
    if not text_a.strip() and not text_b.strip():
        return 1.0
    if not text_a.strip() or not text_b.strip():
        return 0.0

    embs = embedding_model.encode([text_a, text_b])
    sim = cosine_similarity([embs[0]], [embs[1]])[0][0]
    return float(sim)

def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter for semantic metrics."""
    if not isinstance(text, str):
        return []
    parts = re.split(r"[\.\?\!\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def semantic_context_stats(answer: str, context: Optional[str]) -> Dict[str, float]:
    """
    Semantic context precision / recall (Ragas-like).

    Treat **answer sentences** as content and **context sentences** as evidence:
      - context_precision: fraction of answer content supported by context
      - context_recall:    fraction of relevant context used in the answer
    Implemented via sentence embeddings + cosine similarity.
    """
    if not isinstance(context, str) or not context.strip():
        return {"context_precision": 0.0, "context_recall": 0.0}

    ans_sents = _split_sentences(answer)
    ctx_sents = _split_sentences(context)

    if not ans_sents or not ctx_sents:
        return {"context_precision": 0.0, "context_recall": 0.0}

    ans_embs = embedding_model.encode(ans_sents)
    ctx_embs = embedding_model.encode(ctx_sents)

    sim_matrix = cosine_similarity(ans_embs, ctx_embs)  # [n_ans x n_ctx]

    # For each answer sentence, max sim to any context sentence → precision
    ans_to_ctx_max = sim_matrix.max(axis=1)
    # For each context sentence, max sim to any answer sentence → recall
    ctx_to_ans_max = sim_matrix.max(axis=0)

    context_precision = float(ans_to_ctx_max.mean())
    context_recall = float(ctx_to_ans_max.mean())

    return {
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
def answer_context_overlap_stats(answer: str, context: Optional[str]) -> Dict[str, float]:
    """
    String-level overlap between answer and context for hallucination signal.

    - answer_context_overlap: fraction of answer tokens that appear in context.
    - hallucination_rate: 1 - answer_context_overlap.
    """
    if not isinstance(answer, str) or not answer.strip():
        return {"answer_context_overlap": 0.0, "hallucination_rate": 1.0}
    if not isinstance(context, str) or not context.strip():
        return {"answer_context_overlap": 0.0, "hallucination_rate": 1.0}

    ans_tokens = tokenize(answer)
    ctx_tokens = tokenize(context)

    if not ans_tokens or not ctx_tokens:
        return {"answer_context_overlap": 0.0, "hallucination_rate": 1.0}

    ans_counts = {}
    for t in ans_tokens:
        ans_counts[t] = ans_counts.get(t, 0) + 1

    ctx_counts = {}
    for t in ctx_tokens:
        ctx_counts[t] = ctx_counts.get(t, 0) + 1

    overlap = 0
    for t, c in ans_counts.items():
        if t in ctx_counts:
            overlap += min(c, ctx_counts[t])

    answer_context_overlap = overlap / len(ans_tokens)
    hallucination_rate = 1.0 - answer_context_overlap

    return {
        "answer_context_overlap": answer_context_overlap,
        "hallucination_rate": hallucination_rate,
    }

def faithfulness_score(answer: str, context: Optional[str]) -> float:
    """
    Faithfulness (Ragas-style & DeepEval-style):
    how well the answer is supported by the provided context.
    """
    if not isinstance(context, str) or not context.strip():
        return 0.0
    return embedding_similarity(answer, context)

def answer_relevancy_score(answer: str,
                           question: Optional[str],
                           context: Optional[str]) -> float:
    """
    RAGAS-style answer relevancy:
    is the answer appropriate given question + context (combined).
    """
    if not isinstance(answer, str) or not answer.strip():
        return 0.0

    parts = []
    if isinstance(question, str) and question.strip():
        parts.append(question.strip())
    if isinstance(context, str) and context.strip():
        parts.append(context.strip())

    if not parts:
        return 0.0

    qc = " \n".join(parts)
    return embedding_similarity(answer, qc)

def evaluate_pair(pred: str,
                  ref: str,
                  *,
                  question: Optional[str] = None,
                  context: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate a single prediction using DeepEval / Ragas-style metrics.

    Metrics (all roughly 0–1):
      - deepeval_relevance:       answer vs question (semantic)
      - deepeval_faithfulness:    answer vs context (semantic)
      - deepeval_hallucination:   hallucination likelihood (1 - grounded tokens)
      - answer_relevancy:         RAGAS answer relevance (question+context-aware)
      - faithfulness:             RAGAS faithfulness (answer vs context)
      - context_precision:        fraction of answer content supported by context
      - context_recall:           fraction of context used in the answer
      - answer_context_overlap:   fraction of answer tokens in context
      - hallucination_rate:       same as deepeval_hallucination
    """
    # Semantic context precision/recall between answer and context
    ctx_stats = semantic_context_stats(pred, context)

    # String-based grounding / hallucination
    ans_ctx_stats = answer_context_overlap_stats(pred, context)

    # Ragas-style grounding and answer relevancy
    faith = faithfulness_score(pred, context)
    ans_rel_ragas = answer_relevancy_score(pred, question, context)

    # DeepEval-style views
    deepeval_rel = (
        embedding_similarity(pred, question)
        if isinstance(question, str) and question.strip()
        else 0.0
    )
    deepeval_faith = faith
    deepeval_hall = ans_ctx_stats["hallucination_rate"]

    return {
        "deepeval_relevance": deepeval_rel,
        "deepeval_faithfulness": deepeval_faith,
        "deepeval_hallucination": deepeval_hall,
        "answer_relevancy": ans_rel_ragas,
        "faithfulness": faith,
        "context_precision": ctx_stats["context_precision"],
        "context_recall": ctx_stats["context_recall"],
        "answer_context_overlap": ans_ctx_stats["answer_context_overlap"],
        "hallucination_rate": ans_ctx_stats["hallucination_rate"],
    }


df = pd.read_csv(CSV_PATH, encoding = "latin")
print(f"Loaded {len(df)} rows from {CSV_PATH}.\n")

results = []

for col in ANSWER_COLUMNS:
    if col not in df.columns:
        print(f"Warning: column '{col}' not found, skipping.")
        continue

    context_col = CONTEXT_MAPPING.get(col)

    def _eval_row(row: pd.Series) -> Dict[str, float]:
        ctx_val = (
            row[context_col]
            if context_col and context_col in row and isinstance(row[context_col], str)
            else None
        )
        return evaluate_pair(
            pred=row[col],
            ref=row[REFERENCE_COLUMN],
            question=row[QUESTION_COLUMN],
            context=ctx_val,
        )

    metrics_per_row = df.apply(_eval_row, axis=1)
    metrics_df = pd.DataFrame(list(metrics_per_row))

    summary = metrics_df.mean().to_dict()
    summary["model"] = col
    results.append(summary)

metrics_summary_df = pd.DataFrame(results)[[
    "model",
    "deepeval_relevance",
    "deepeval_faithfulness",
    "deepeval_hallucination",
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_context_overlap",
    "hallucination_rate",
]]

print("Metric summary per answer type:\n")
print(metrics_summary_df.to_string(index=False))


if not metrics_summary_df.empty:
    plot_df = metrics_summary_df.set_index("model")
    plt.figure(figsize=(10, 6))
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="Blues")
    plt.title("RAG Evaluation Metrics per Method")
    plt.tight_layout()
    plt.show()
