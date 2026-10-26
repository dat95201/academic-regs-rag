"""Hybrid retrieval: dense (bge-m3) + sparse (BM25) fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.rag.embeddings import OllamaEmbedder, bm25_query, nfc
from app.rag.store import QdrantStore


@dataclass
class Hit:
    id: str
    score: float  # fusion score (RRF when hybrid, cosine otherwise)
    dense_score: float  # cosine similarity, comparable to MIN_RELEVANCE_SCORE
    payload: dict


class Retriever:
    def __init__(self, settings: Settings, embedder: OllamaEmbedder, store: QdrantStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def search(self, query: str, limit: int | None = None) -> list[Hit]:
        query = nfc(query).strip()
        limit = limit or self.settings.top_k_retrieve
        dense = self.embedder.embed([query])[0]
        sparse = bm25_query(query) if self.settings.hybrid_search else None
        points = self.store.search(dense, sparse, limit)
        cosines = self.store.dense_scores(dense, [str(p.id) for p in points])
        return [
            Hit(str(p.id), p.score, cosines.get(str(p.id), 0.0), p.payload or {}) for p in points
        ]

    def relevant(self, query: str) -> list[Hit]:
        """Top TOP_K_FINAL hits, dropping everything when nothing reaches MIN_RELEVANCE_SCORE."""
        hits = self.search(query)
        if not any(h.dense_score >= self.settings.min_relevance_score for h in hits):
            return []
        # TODO(Task 5): rerank with RERANK_MODEL when RERANK_ENABLED.
        return hits[: self.settings.top_k_final]
