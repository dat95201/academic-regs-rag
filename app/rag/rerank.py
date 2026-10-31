"""Reranking without a cross-encoder.

A cross-encoder (bge-reranker) is the usual answer, but it means torch and a second model on
a machine that is already running an LLM and an embedder on CPU. Two cheaper rerankers:

* `heuristic` (default, free): re-scores the candidates with signals the vector search throws
  away - how many of the query's own terms actually occur in the chunk, whether they occur in
  the article's title, whether the question names the article by number - and spreads the
  result across articles so five near-identical pieces of one article do not fill the answer.
* `llm` (opt-in): one extra call to the chat model with only the candidate headings, asking it
  to order them. Costs a round trip; useful when the questions are vague.

Term weights are a local IDF: a term in every candidate ("sinh viên") says nothing, a term in
one or two ("phúc tra") says everything. No corpus statistics needed.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from typing import Protocol

from app.rag.citations import Ref

log = logging.getLogger("rerank")

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Function words carry no retrieval signal in Vietnamese questions.
STOPWORDS = {
    "là", "của", "và", "có", "cho", "các", "được", "khi", "thì", "trong", "với", "này", "đó",
    "những", "một", "bao", "nhiêu", "nào", "gì", "sao", "không", "ạ", "ở", "từ", "đến", "về",
    "theo", "như", "thế", "mà", "em", "mình", "tôi", "hỏi", "cần", "phải", "sẽ", "bị",
    "nếu", "hay", "hoặc", "ra", "vào", "lại", "đi", "muốn", "ai", "do", "vì", "nên", "rồi",
    # Deictics: they point back at the conversation instead of naming anything in the corpus.
    "vậy", "nó", "kia", "ấy", "họ", "chúng", "còn",
}

# How much each signal moves the final score. Dense similarity stays dominant; the rest break
# ties between chunks the embedder finds equally close.
W_RETRIEVAL = 1.00  # fused dense+sparse score from Qdrant
W_LEXICAL = 0.45  # query terms present in the chunk body
W_TITLE = 0.35  # query terms present in the article / section title
W_REF = 1.20  # the question names this article by number
DIVERSITY_PENALTY = 0.18  # grows with each extra chunk already taken from the same article


class Completer(Protocol):
    def complete(self, messages: list[dict], temperature: float | None = ...) -> str: ...


def terms(text: str) -> list[str]:
    """Content syllables plus syllable bigrams, matching how the BM25 index is built."""
    words = [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


def weights(query_terms: Sequence[str], documents: Sequence[str]) -> dict[str, float]:
    """Local IDF: rare among the candidates = informative."""
    n = max(len(documents), 1)
    lowered = [d.lower() for d in documents]
    out: dict[str, float] = {}
    for t in set(query_terms):
        df = sum(t in d for d in lowered)
        out[t] = math.log(1 + n / (1 + df))
    return out


def coverage(query_terms: Sequence[str], weight: dict[str, float], text: str) -> float:
    """Share of the query's weight that occurs in `text`, in [0, 1]."""
    total = sum(weight.get(t, 0.0) for t in set(query_terms))
    if total <= 0:
        return 0.0
    low = text.lower()
    hit = sum(w for t, w in weight.items() if t in query_terms and t in low)
    return hit / total


def _normalize(values: Sequence[float]) -> list[float]:
    """Scale by the best score, not min-max.

    Min-max would map the top and bottom of any list to 1 and 0, so two candidates scoring
    0.51 and 0.50 would look as far apart as 1.0 and 0.0 and no other signal could ever move
    them. Dividing by the maximum keeps "these two are nearly tied" meaningful.
    """
    hi = max(values, default=0.0)
    if hi <= 1e-9:
        return [0.0] * len(values)
    return [max(v, 0.0) / hi for v in values]


def _title_of(payload: dict) -> str:
    return " ".join(
        str(payload.get(k) or "")
        for k in ("article", "article_title", "chapter_title", "doc_short_title")
    ) + " " + " ".join(payload.get("sections") or [])


def heuristic(query: str, hits: list, refs: Sequence[Ref] = (), top_k: int | None = None) -> list:
    """Re-score and re-order `hits` (anything with .score/.payload). Pure function, no I/O."""
    if not hits:
        return []
    query_terms = terms(query)
    bodies = [str(h.payload.get("text", "")) for h in hits]
    weight = weights(query_terms, bodies)
    named = {r.article for r in refs}
    base = _normalize([h.score for h in hits])

    scored = []
    for h, retrieval in zip(hits, base, strict=True):
        payload = h.payload
        score = (
            W_RETRIEVAL * retrieval
            + W_LEXICAL * coverage(query_terms, weight, payload.get("text", ""))
            + W_TITLE * coverage(query_terms, weight, _title_of(payload))
            + W_REF * (1.0 if payload.get("article") in named else 0.0)
        )
        scored.append((score, h))
    scored.sort(key=lambda s: s[0], reverse=True)

    # Spread the result across articles. The penalty is cumulative (0.18, then 0.54, then
    # 1.08), so a second passage of the best article still competes while a fourth has to be
    # outstanding to beat the first passage of another article.
    out, seen = [], {}
    while scored and (top_k is None or len(out) < top_k):
        best_i, best_value = 0, -1e9
        for i, (score, h) in enumerate(scored):
            taken = seen.get(h.payload.get("article"), 0)
            value = score - DIVERSITY_PENALTY * taken * (taken + 1) / 2
            if value > best_value:
                best_i, best_value = i, value
        _, hit = scored.pop(best_i)
        seen[hit.payload.get("article")] = seen.get(hit.payload.get("article"), 0) + 1
        hit.rerank_score = round(best_value, 4)
        out.append(hit)
    return out


PROMPT = (
    "Bạn sắp xếp mức liên quan của các trích đoạn quy chế với câu hỏi. "
    "Chỉ trả về một mảng JSON các số thứ tự, từ liên quan nhất đến ít nhất, "
    "ví dụ [3,1,5]. Không giải thích."
)


def llm(client: Completer, query: str, hits: list, top_k: int) -> list:
    """Listwise rerank with the chat model: headings only, so the prompt stays small."""
    if len(hits) <= 1:
        return hits[:top_k]
    listing = "\n".join(
        f"{i + 1}. {h.payload.get('article')} {h.payload.get('article_title') or ''} :: "
        f"{str(h.payload.get('text', ''))[:220]}"
        for i, h in enumerate(hits)
    )
    try:
        raw = client.complete(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"CÂU HỎI: {query}\n\nTRÍCH ĐOẠN:\n{listing}"},
            ],
            temperature=0,
        )
        order = json.loads(re.search(r"\[[^\]]*\]", raw).group(0))
        ranked = [hits[i - 1] for i in order if isinstance(i, int) and 1 <= i <= len(hits)]
    except Exception:  # noqa: BLE001 - any parse/transport problem falls back to the input order
        log.warning("llm rerank failed, keeping retrieval order")
        return hits[:top_k]
    ranked += [h for h in hits if h not in ranked]
    return ranked[:top_k]
