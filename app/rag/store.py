"""Qdrant collection holding the chunks: dense `bge-m3` vectors plus optional BM25 sparse vectors."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from qdrant_client import QdrantClient, models

log = logging.getLogger("store")

DENSE = "dense"
SPARSE = "bm25"


def connect(url: str) -> QdrantClient:
    # ":memory:" or a directory path gives an embedded Qdrant (used by tests / offline runs).
    if url.startswith("http"):
        return QdrantClient(url=url)
    if url == ":memory:":
        return QdrantClient(location=url)
    return QdrantClient(path=url)


class QdrantStore:
    def __init__(self, url: str, collection: str, client: QdrantClient | None = None):
        self.client = client or connect(url)
        self.collection = collection

    # --- schema ---

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def mismatch(self, dim: int, hybrid: bool, embed_model: str) -> str | None:
        """Why the existing collection cannot be reused, or None if it can."""
        info = self.client.get_collection(self.collection)
        dense = info.config.params.vectors
        size = dense[DENSE].size if isinstance(dense, dict) and DENSE in dense else None
        if size != dim:
            return f"dense vector size {size} != {dim}"
        has_sparse = bool(info.config.params.sparse_vectors) and SPARSE in info.config.params.sparse_vectors
        if has_sparse != hybrid:
            return f"hybrid search changed (sparse vectors present: {has_sparse})"
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=["embed_model"])
        if points and points[0].payload.get("embed_model") != embed_model:
            return f"embedded with {points[0].payload.get('embed_model')}, now {embed_model}"
        return None

    def ensure(self, dim: int, hybrid: bool, embed_model: str) -> bool:
        """Create the collection, or recreate it if the embedding setup changed.
        Returns True when the collection is new (everything must be ingested)."""
        if self.exists():
            reason = self.mismatch(dim, hybrid, embed_model)
            if reason is None:
                return False
            log.warning("recreating collection %s: %s", self.collection, reason)
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config=(
                {SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)} if hybrid else None
            ),
        )
        self.client.create_payload_index(self.collection, "doc_id", models.PayloadSchemaType.KEYWORD)
        log.info("created collection %s (dim=%d, hybrid=%s)", self.collection, dim, hybrid)
        return True

    # --- data ---

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
            wait=True,
        )

    def upsert(self, points: Iterable[models.PointStruct], batch_size: int = 64) -> None:
        batch: list[models.PointStruct] = []
        for p in points:
            batch.append(p)
            if len(batch) == batch_size:
                self.client.upsert(self.collection, batch, wait=True)
                batch = []
        if batch:
            self.client.upsert(self.collection, batch, wait=True)

    # --- search ---

    def search(
        self,
        dense: list[float],
        sparse: tuple[list[int], list[float]] | None,
        limit: int,
    ) -> list[models.ScoredPoint]:
        if sparse is None:
            return self.client.query_points(
                self.collection, query=dense, using=DENSE, limit=limit, with_payload=True
            ).points
        indices, values = sparse
        return self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=limit),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

    def dense_scores(self, dense: list[float], ids: list[str]) -> dict[str, float]:
        """Cosine similarity of the query to specific points (RRF scores are rank-based)."""
        if not ids:
            return {}
        hits = self.client.query_points(
            self.collection,
            query=dense,
            using=DENSE,
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)]),
            limit=len(ids),
        ).points
        return {str(h.id): h.score for h in hits}


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )
