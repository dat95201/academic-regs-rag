"""Retrieval: expand the query, search both indexes, fuse, rerank, widen.

    question
      -> citations.detect   "Điều 15"/"Phụ lục 7" named in the question
      -> glossary.expand    "GPA", "rớt môn" -> the wording the regulation uses
      -> one batched Qdrant call: dense (bge-m3) + sparse (BM25)
      -> weighted RRF, keeping the real cosine so the relevance gate still means something
      -> exact article lookup merged in on top, when the question named one
      -> rerank (heuristic by default, see app.rag.rerank)
      -> widen each kept chunk with its neighbours in the same article

Everything but the Qdrant call and the embedding is pure CPU, so the whole pipeline costs one
round trip to the vector store and one to the embedder (cached for repeated questions).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.rag import citations, rerank
from app.rag.embeddings import OllamaEmbedder, bm25_query, nfc
from app.rag.glossary import Expansion, expand
from app.rag.store import QdrantStore

log = logging.getLogger("retriever")

# Reciprocal-rank-fusion constant. 60 is the value from the original paper and is not
# sensitive; it only decides how fast the contribution of a rank decays.
RRF_K = 60


@dataclass
class Hit:
    id: str
    score: float  # fused score across the dense and sparse branches
    dense_score: float  # cosine similarity, comparable to MIN_RELEVANCE_SCORE
    sparse_score: float  # BM25 score, 0 when the chunk was not in the sparse branch
    payload: dict
    origin: str = "hybrid"  # hybrid | article | context
    rerank_score: float | None = None

    def view(self) -> dict:
        return {
            "id": self.id,
            "score": round(self.score, 4),
            "cosine": round(self.dense_score, 4),
            "bm25": round(self.sparse_score, 4),
            "rerank": self.rerank_score,
            "origin": self.origin,
            "article": self.payload.get("article"),
            "article_title": self.payload.get("article_title"),
            "sections": self.payload.get("sections"),
            "page": self.payload.get("page"),
        }


@dataclass
class Retrieval:
    """Result plus everything the debug page and the query log want to show."""

    hits: list[Hit] = field(default_factory=list)
    query: str = ""
    expansion: Expansion | None = None
    refs: list[citations.Ref] = field(default_factory=list)
    candidates: int = 0
    took_ms: int = 0
    gated: bool = False  # nothing passed MIN_RELEVANCE_SCORE
    # Best candidates even when the gate rejected them, so the UI can suggest a direction
    # instead of only saying no.
    fallback: list[Hit] = field(default_factory=list)

    @property
    def payloads(self) -> list[dict]:
        return [h.payload for h in self.hits]


def fuse(dense_hits, sparse_hits, w_dense: float, w_sparse: float) -> list[Hit]:
    """Weighted RRF over the two branches, keeping each branch's own score."""
    hits: dict[str, Hit] = {}
    fused: dict[str, float] = {}
    for points, weight, attr in ((dense_hits, w_dense, "dense_score"),
                                 (sparse_hits, w_sparse, "sparse_score")):
        for rank, point in enumerate(points):
            key = str(point.id)
            hit = hits.get(key)
            if hit is None:
                hit = hits[key] = Hit(key, 0.0, 0.0, 0.0, point.payload or {})
            setattr(hit, attr, float(point.score))
            fused[key] = fused.get(key, 0.0) + weight / (RRF_K + rank + 1)
    for key, score in fused.items():
        hits[key].score = score
    return sorted(hits.values(), key=lambda h: h.score, reverse=True)


class Retriever:
    def __init__(self, settings: Settings, embedder: OllamaEmbedder, store: QdrantStore,
                 llm=None):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm  # only used when RERANK_MODE=llm

    # --- steps ---

    def _article_hits(self, refs: list[citations.Ref], known: set[str]) -> list[Hit]:
        """Chunks of an article the question named by number, which vector search can miss."""
        records = self.store.by_article([r.article for r in refs], limit=self.settings.top_k_final * 3)
        return [
            Hit(str(r.id), 0.0, 0.0, 0.0, r.payload or {}, origin="article")
            for r in records
            if str(r.id) not in known
        ]

    def _select(self, hits: list[Hit], named: set[str]) -> list[Hit]:
        """Keep the chunks that are actually relevant, not a fixed quota of them.

        The relevance score used to be a single gate: if *any* candidate cleared
        MIN_RELEVANCE_SCORE, the whole top-k went to the model and to the reader. So a
        question answered by one article still produced five "sources", four of them padding,
        and a question nothing could answer still produced five - right up until the model
        said it found nothing, contradicting the list sitting underneath it.

        Now every chunk has to clear the bar itself: an absolute floor, and a relative one
        tied to the best chunk of this particular search. Two carve-outs: a chunk fetched
        because the question named its article is kept whatever its score, and a chunk found
        only by BM25 has no cosine to be judged by, so its rank is all we have.
        """
        settings = self.settings
        best = max((h.dense_score for h in hits), default=0.0)
        floor = max(settings.min_relevance_score, best * settings.relative_cutoff)
        kept = [
            h for h in hits
            if h.origin == "article"
            or h.payload.get("article") in named
            or h.dense_score <= 0.0  # sparse-only hit: no cosine, judged by its rank alone
            or h.dense_score >= floor
        ]
        return kept[: settings.top_k_final]

    def _widen(self, hits: list[Hit]) -> list[Hit]:
        """Add the neighbouring chunks of each kept article, as background context.

        A regulation answer often sits across '15.3' and '15.4'; retrieval returns one of them.
        The neighbours go last, so `build_context` drops them first if the budget is tight.
        """
        budget = min(self.settings.context_neighbours,
                     max(0, self.settings.top_k_final - len(hits)))
        if budget <= 0:
            return hits
        taken = {h.id for h in hits}
        extra: list[Hit] = []
        for hit in list(hits):
            if len(extra) >= budget:
                break
            for record in self.store.neighbours(
                hit.payload.get("doc_id", ""), hit.payload.get("article", ""),
                taken, limit=budget,
            ):
                if str(record.id) in taken:
                    continue
                taken.add(str(record.id))
                extra.append(Hit(str(record.id), 0.0, 0.0, 0.0, record.payload or {},
                                 origin="context"))
                if len(extra) >= budget:
                    break
        return hits + extra

    # --- entry points ---

    def retrieve(self, query: str, limit: int | None = None) -> Retrieval:
        started = time.monotonic()
        settings = self.settings
        query = nfc(query).strip()
        limit = limit or settings.top_k_retrieve

        refs = citations.detect(query) if settings.article_lookup else []
        expansion = expand(query, settings.glossary_expansion)
        dense_text = expansion.text if settings.glossary_for_dense else query

        dense_vector = self.embedder.embed_query(dense_text)
        sparse = bm25_query(expansion.text) if settings.hybrid_search else None
        dense_hits, sparse_hits = self.store.search(dense_vector, sparse, limit)
        hits = fuse(dense_hits, sparse_hits, settings.dense_weight, settings.sparse_weight)

        if refs:
            hits += self._article_hits(refs, {h.id for h in hits})

        out = Retrieval(hits=hits, query=query, expansion=expansion, refs=refs,
                        candidates=len(hits))
        out.took_ms = int((time.monotonic() - started) * 1000)
        return out

    def search(self, query: str, limit: int | None = None) -> list[Hit]:
        """Raw fused candidates - used by /search and the debug page."""
        return self.retrieve(query, limit).hits

    def relevant(self, query: str) -> Retrieval:
        """The chunks worth showing the model: gated, reranked, widened."""
        result = self.retrieve(query)
        settings = self.settings
        named = {r.article for r in result.refs}
        hits = result.hits
        mode = settings.rerank_mode if settings.rerank_enabled else "off"
        if mode == "llm" and self.llm is not None:
            hits = rerank.llm(self.llm, query, hits[: settings.rerank_candidates],
                              settings.top_k_final)
        elif mode == "heuristic":
            hits = rerank.heuristic(result.expansion.text if result.expansion else query,
                                    hits[: settings.rerank_candidates], result.refs,
                                    settings.top_k_final)
        hits = self._select(hits[: settings.top_k_final], named)
        if not hits:
            result.gated = True
            result.fallback = result.hits[:3]
            result.hits = []
            return result

        if settings.context_expansion and settings.context_neighbours > 0:
            hits = self._widen(hits)
        result.hits = hits
        return result
