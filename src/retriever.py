"""
Unified Retrieval Module — Hybrid Search with RRF.

Provides a single `retrieve()` entry point that:
  1. Runs BM25 (keyword) + Vector (semantic) search
  2. Fuses results using Reciprocal Rank Fusion (RRF)
  3. Returns a ranked, deduplicated list of documents with scores
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """Bundle returned by retrieve()."""
    docs: list[Document] = field(default_factory=list)
    rrf_scores: dict[int, float] = field(default_factory=dict)  # idx → score
    best_vector_score: float = 0.0


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Merge multiple ranked lists using RRF.
    Returns docs sorted by descending fused score.
    k = smoothing constant (standard default 60).
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)
    return [(doc_map[k], doc_scores[k]) for k in sorted_keys]


def _doc_key(doc: Document) -> str:
    """Stable dedup key for a document chunk."""
    src = (doc.metadata.get("source") or "")
    title = (doc.metadata.get("title") or "")
    snippet = (doc.page_content or "")[:150]
    return f"{src}|{title}|{snippet}"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_chunks_from_file(path: Path) -> list[Document]:
    """Load Document objects from a JSONL file (for BM25)."""
    docs = []
    if not path.exists():
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs.append(
                Document(
                    page_content=obj.get("page_content", "") or "",
                    metadata=obj.get("metadata", {}) or {},
                )
            )
    return docs


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------
def retrieve(
    question: str,
    vectorstore: Chroma,
    bm25_retriever: BM25Retriever | None,
    k: int = 5,
) -> RetrievalResult:
    """
    Single entry point for retrieval.

    1. BM25 top-k + Vector top-k (with scores)
    2. RRF fusion across both lists
    3. Return top-k results + metadata
    """
    ranked_lists: list[list[Document]] = []

    # Vector search with scores (single call — gives us both docs and scores)
    best_vector_score = 0.0
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(question, k=k)
        if scored:
            best_vector_score = max(score for _, score in scored)
            ranked_lists.append([doc for doc, _ in scored])
    except Exception:
        vec_docs = vectorstore.similarity_search(question, k=k)
        ranked_lists.append(vec_docs)

    # BM25 search
    if bm25_retriever is not None:
        bm25_docs = bm25_retriever.invoke(question)[:k]
        ranked_lists.append(bm25_docs)

    # RRF fusion
    fused = reciprocal_rank_fusion(ranked_lists)
    top_docs = fused[:k]

    return RetrievalResult(
        docs=[doc for doc, _ in top_docs],
        rrf_scores={i: score for i, (_, score) in enumerate(top_docs)},
        best_vector_score=best_vector_score,
    )
