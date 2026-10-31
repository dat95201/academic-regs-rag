"""Qdrant collection holding the chunks: dense `bge-m3` vectors plus optional BM25 sparse vectors.

Searching returns the two branches separately (one batched round trip) instead of asking Qdrant
to fuse them: RRF throws the scores away and keeps only the ranks, and the scores are what the
relevance gate and the reranker need. Fusion happens in `app.rag.retriever`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from qdrant_client import QdrantClient, models

log = logging.getLogger("store")

DENSE = "dense"
SPARSE = "bm25"
# Payload fields that get an index: filtered on every exact-article lookup and every
# neighbour expansion, so an index turns a full scan into a lookup.
INDEXED_FIELDS = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "article": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
}


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

    def ensure_indexes(self) -> None:
        """Create the payload indexes if they are missing. Safe to call on every start."""
        for field, schema in INDEXED_FIELDS.items():
            try:
                self.client.create_payload_index(self.collection, field, schema)
            except Exception as e:  # noqa: BLE001 - already indexed, or a local client without indexes
                log.debug("payload index %s: %s", field, e)

    def ensure(self, dim: int, hybrid: bool, embed_model: str) -> bool:
        """Create the collection, or recreate it if the embedding setup changed.
        Returns True when the collection is new (everything must be ingested)."""
        if self.exists():
            reason = self.mismatch(dim, hybrid, embed_model)
            if reason is None:
                self.ensure_indexes()
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
        self.ensure_indexes()
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
            points_selector=models.FilterSelector(filter=_match(doc_id=doc_id)),
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
    ) -> tuple[list[models.ScoredPoint], list[models.ScoredPoint]]:
        """(dense hits, sparse hits) in one round trip. Sparse is empty when hybrid is off."""
        if not indices_present(sparse):
            hits = self.client.query_points(
                self.collection, query=dense, using=DENSE, limit=limit, with_payload=True
            ).points
            return hits, []
        indices, values = sparse
        responses = self.client.query_batch_points(
            self.collection,
            requests=[
                models.QueryRequest(query=dense, using=DENSE, limit=limit, with_payload=True),
                models.QueryRequest(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE,
                    limit=limit,
                    with_payload=True,
                ),
            ],
        )
        return responses[0].points, responses[1].points

    def by_article(self, articles: Sequence[str], limit: int = 20) -> list[models.Record]:
        """Every chunk of the named articles / appendices ('Điều 15', 'Phụ lục 7')."""
        if not articles:
            return []
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="article",
                                            match=models.MatchAny(any=list(articles)))]
            ),
            limit=limit,
            with_payload=True,
        )
        return sorted(points, key=lambda p: p.payload.get("chunk_index", 0))

    def neighbours(self, doc_id: str, article: str, exclude: set[str], limit: int = 4
                   ) -> list[models.Record]:
        """The other chunks of the same article, for widening a hit into its full context."""
        if not article:
            return []
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=_match(doc_id=doc_id, article=article),
            limit=limit + len(exclude),
            with_payload=True,
        )
        rest = [p for p in points if str(p.id) not in exclude]
        return sorted(rest, key=lambda p: p.payload.get("chunk_index", 0))[:limit]


def indices_present(sparse: tuple[list[int], list[float]] | None) -> bool:
    return bool(sparse and sparse[0])


def _match(**fields: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
              for k, v in fields.items()]
    )
