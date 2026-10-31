"""Dense embeddings via Ollama and BM25-style sparse vectors for hybrid search."""

from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from collections import Counter, OrderedDict

import httpx

from app.config import Settings


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class OllamaEmbedder:
    """Calls Ollama `POST /api/embed` in batches.

    Query vectors are cached: a class of students asks the same handful of questions, and on
    CPU an embedding call is tens of milliseconds that need not be spent twice.
    """

    def __init__(self, settings: Settings):
        self.model = settings.embed_model
        self.batch_size = settings.embed_batch_size
        self.client = httpx.Client(base_url=settings.ollama_url, timeout=settings.ollama_timeout_s)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_max = max(0, settings.query_cache_size)
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [nfc(t) for t in texts[i:i + self.batch_size]]
            resp = self.client.post("/api/embed", json={"model": self.model, "input": batch})
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Ollama has no model '{self.model}'. Run: ollama pull {self.model}"
                )
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        key = nfc(text).strip()
        if self._cache_max:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    self.cache_hits += 1
                    return self._cache[key]
        vector = self.embed([key])[0]
        if self._cache_max:
            with self._lock:
                self.cache_misses += 1
                self._cache[key] = vector
                while len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
        return vector

    def dimension(self) -> int:
        return len(self.embed(["dimension probe"])[0])


# --- sparse (BM25) ---
#
# fastembed's Qdrant/bm25 model stems and drops stopwords for English, which mangles
# Vietnamese. Vietnamese words are mostly 1–3 syllables ("học phần", "tín chỉ"), so syllables
# plus syllable bigrams are indexed. The vector holds BM25 term-frequency weights; Qdrant
# applies IDF itself (sparse vector config `modifier=IDF`).

TOKEN_RE = re.compile(r"\w+")
BM25_K1 = 1.2
BM25_B = 0.75
# Fallback when the corpus average is unknown; ingest measures the real one (see run.py).
BM25_AVG_LEN = 400.0


def _terms(text: str) -> tuple[list[str], int]:
    words = TOKEN_RE.findall(nfc(text).lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)], len(words)


def _index(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF


def _merge(pairs: dict[str, float]) -> tuple[list[int], list[float]]:
    merged: dict[int, float] = {}
    for term, weight in pairs.items():
        idx = _index(term)
        merged[idx] = merged.get(idx, 0.0) + weight  # hash collisions just add up
    return list(merged), list(merged.values())


def corpus_avg_len(texts: list[str]) -> float:
    """Mean syllable count of the chunks, the `avgdl` of the BM25 formula."""
    lengths = [len(TOKEN_RE.findall(nfc(t).lower())) for t in texts]
    return sum(lengths) / len(lengths) if lengths else BM25_AVG_LEN


def bm25_document(text: str, avg_len: float = BM25_AVG_LEN) -> tuple[list[int], list[float]]:
    terms, length = _terms(text)
    norm = BM25_K1 * (1 - BM25_B + BM25_B * length / max(avg_len, 1.0))
    weights = {t: tf * (BM25_K1 + 1) / (tf + norm) for t, tf in Counter(terms).items()}
    return _merge(weights)


def bm25_query(text: str) -> tuple[list[int], list[float]]:
    terms, _ = _terms(text)
    return _merge({t: 1.0 for t in set(terms)})
