"""Dense embeddings via Ollama and BM25-style sparse vectors for hybrid search."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

import httpx

from app.config import Settings


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class OllamaEmbedder:
    """Calls Ollama `POST /api/embed` in batches."""

    def __init__(self, settings: Settings):
        self.model = settings.embed_model
        self.batch_size = settings.embed_batch_size
        self.client = httpx.Client(base_url=settings.ollama_url, timeout=settings.ollama_timeout_s)

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
BM25_AVG_LEN = 400  # syllables in a typical chunk (~500 tokens)


def _terms(text: str) -> tuple[list[str], int]:
    words = TOKEN_RE.findall(nfc(text).lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])], len(words)


def _index(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF


def _merge(pairs: dict[str, float]) -> tuple[list[int], list[float]]:
    merged: dict[int, float] = {}
    for term, weight in pairs.items():
        idx = _index(term)
        merged[idx] = merged.get(idx, 0.0) + weight  # hash collisions just add up
    return list(merged), list(merged.values())


def bm25_document(text: str) -> tuple[list[int], list[float]]:
    terms, length = _terms(text)
    norm = BM25_K1 * (1 - BM25_B + BM25_B * length / BM25_AVG_LEN)
    weights = {t: tf * (BM25_K1 + 1) / (tf + norm) for t, tf in Counter(terms).items()}
    return _merge(weights)


def bm25_query(text: str) -> tuple[list[int], list[float]]:
    terms, _ = _terms(text)
    return _merge({t: 1.0 for t in set(terms)})

