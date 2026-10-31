"""Fixtures: a real in-memory Qdrant seeded from the real document, and a fake embedder.

The embedder is deterministic hashing bag-of-words rather than bge-m3: the tests check the
plumbing (fusion, gating, exact lookup, widening, event order), not the model. That keeps the
suite runnable with no Ollama, no network and no GPU - which is the only way it gets run.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.ingest.chunk import chunk_file
from app.ingest.run import to_points
from app.rag.embeddings import corpus_avg_len
from app.rag.retriever import Retriever
from app.rag.store import QdrantStore

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DIM = 96
TOKEN_RE = re.compile(r"\w+")


class HashEmbedder:
    """Hashing bag-of-words, L2-normalised: cosine then tracks lexical overlap."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.cache_hits = 0
        self.cache_misses = 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in TOKEN_RE.findall(text.lower()):
            h = hashlib.blake2b(word.encode(), digest_size=4).digest()
            vec[int.from_bytes(h, "little") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.cache_misses += 1
        return self._vector(text)

    def dimension(self) -> int:
        return self.dim


class FakeLLM:
    """Records what it was asked and replays canned answers."""

    def __init__(self, reply: str = "Trả lời [Quy định, Điều 5]", chunks: list[str] | None = None):
        self.reply = reply
        self.chunks = chunks
        self.completions: list[list[dict]] = []
        self.streams: list[list[dict]] = []

    def complete(self, messages, temperature=None):
        self.completions.append(messages)
        return self.reply

    def stream(self, messages, temperature=None):
        self.streams.append(messages)
        yield from (self.chunks if self.chunks is not None else [self.reply])


@pytest.fixture(scope="session")
def markdown() -> Path:
    files = sorted(p for p in PROCESSED.glob("*.md") if ".pages-" not in p.name)
    if not files:
        pytest.skip("no reviewed Markdown in data/processed")
    return files[0]


@pytest.fixture(scope="session")
def chunks(markdown: Path):
    return chunk_file(markdown, 800, 100)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(qdrant_url=":memory:", collection_name="test", hybrid_search=True,
                    top_k_retrieve=20, top_k_final=5, min_relevance_score=0.05,
                    context_neighbours=2, log_queries=False)


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture(scope="session")
def store(settings: Settings, embedder: HashEmbedder, chunks) -> QdrantStore:
    store = QdrantStore(":memory:", settings.collection_name)
    store.ensure(embedder.dimension(), settings.hybrid_search, settings.embed_model)
    avg = corpus_avg_len([c.embed_text for c in chunks])
    vectors = embedder.embed([c.embed_text for c in chunks])
    store.upsert(to_points(chunks, vectors, settings, avg))
    return store


@pytest.fixture
def retriever(settings: Settings, embedder: HashEmbedder, store: QdrantStore) -> Retriever:
    return Retriever(settings, embedder, store)
