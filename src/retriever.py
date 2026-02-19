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


def load_bm25_retriever(path: Path) -> BM25Retriever | None:
    """Load pre-built BM25 retriever from pickle."""
    if not path.exists():
        return None
    try:
        import dill
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        print(f"BM25 Save Load Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Query Expansion
# ---------------------------------------------------------------------------
def generate_multi_queries(llm, original_query: str) -> list[str]:
    """Generate 3 variations of the original query for better recall."""
    prompt = (
        "You are an AI language model assistant. Your task is to generate 3 different versions "
        "of the given user question to retrieve relevant documents from a vector database. "
        "By generating multiple perspectives on the user question, your goal is to help "
        "the user overcome some of the limitations of the distance-based similarity search. "
        "Provide these alternative questions separated by newlines. "
        "Original question: " + original_query
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        questions = [q.strip() for q in content.split("\n") if q.strip()]
        return [original_query] + questions[:3]  # original + up to 3 variations
    except Exception:
        return [original_query]


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------
def retrieve(
    question: str,
    vectorstore: Chroma,
    bm25_retriever: BM25Retriever | None,
    llm=None, # Optional LLM for query expansion
    k: int = 5,
) -> RetrievalResult:
    """
    Single entry point for retrieval with optional Multi-Query Expansion.

    1. Generate query variations (if llm provided)
    2. Run BM25 + Vector search for ALL variations
    3. Fuse results using RRF
    4. Return top-k results
    """
    
    # 1. Query Expansion
    queries = [question]
    if llm:
        queries = generate_multi_queries(llm, question)
        # print(f"🔍 Extended Queries: {queries}")

    ranked_lists: list[list[Document]] = []
    best_vector_score = 0.0

    for q in queries:
        # Vector search
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(q, k=k)
            if scored:
                current_best = max(score for _, score in scored)
                best_vector_score = max(best_vector_score, current_best)
                ranked_lists.append([doc for doc, _ in scored])
        except Exception:
            vec_docs = vectorstore.similarity_search(q, k=k)
            ranked_lists.append(vec_docs)

        # BM25 search
        if bm25_retriever is not None:
            bm25_docs = bm25_retriever.invoke(q)[:k]
            ranked_lists.append(bm25_docs)

    # RRF fusion
    fused = reciprocal_rank_fusion(ranked_lists)
    top_docs = fused[:k*2] # Get a bit more to filter duplicates/low quality if needed

    # Deduplicate by content to be safe (RRF already does key-based dedup, but let's be sure)
    unique_docs = []
    seen_content = set()
    rrf_scores = {}
    
    for i, (doc, score) in enumerate(top_docs):
        # Use a hash of the content to deduplicate exact text matches
        content_hash = hash(doc.page_content.strip())
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
            rrf_scores[len(unique_docs)-1] = score # Map new index to score
            
        if len(unique_docs) >= k:
            break

    return RetrievalResult(
        docs=unique_docs,
        rrf_scores=rrf_scores,
        best_vector_score=best_vector_score,
    )