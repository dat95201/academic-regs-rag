"""One chat turn: rewrite follow-up -> retrieve -> answer from excerpts, as a stream of events.

Events, in order:

    query      str    the follow-up rewritten into a standalone question (only when rewritten)
    retrieved  dict   the excerpts, as soon as they are found - before the model starts writing
    token      str    answer text, streamed
    sources    list   the excerpts the answer is based on ([] when the model found nothing)
    done       dict   found, timings, and any citation the excerpts do not support

`retrieved` arrives seconds before the first token on a CPU-only machine, which is the point:
the reader can start opening the source passages while the answer is still being written.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

from app.config import Settings
from app.rag import citations
from app.rag.embeddings import nfc
from app.rag.llm import LLMClient
from app.rag.prompt import NOT_FOUND, answer_messages, build_context, label, rewrite_messages
from app.rag.rerank import STOPWORDS
from app.rag.retriever import Retrieval, Retriever

log = logging.getLogger("chat")

NO_MATCH = (
    "Không tìm thấy nội dung liên quan trong tài liệu hiện có. "
    "Bạn thử hỏi cụ thể hơn (ví dụ nêu rõ học phần, học kỳ, loại chương trình)."
)

# A follow-up only needs the rewrite round trip when it leans on the previous turn: a pronoun,
# a bare continuation ("còn vừa làm vừa học thì sao?"), or too few words to stand alone.
DEPENDENT = re.compile(
    r"(?<!\w)(nó|vậy|thế|đó|này|kia|trên|ấy|còn|thì sao|như vậy|trường hợp (này|đó)|"
    r"cái (này|đó)|họ|chúng)(?!\w)",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\w+", re.UNICODE)
# A rewrite several times longer than the question is no longer a rewrite of it.
REWRITE_MAX_GROWTH = 4


def needs_rewrite(question: str, history: list[dict], enabled: bool = True) -> bool:
    if not history or not enabled:
        return False
    return bool(DEPENDENT.search(question)) or len(question.split()) < 5


def _content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS}


def accept_rewrite(question: str, rewritten: str) -> bool:
    """Whether a rewritten question is still a rewrite of the question that was asked.

    Handed a conversation about one topic and an unrelated new question, a small model will
    sometimes return the *previous* question instead of rewriting the new one. The user asks
    for a hotpot recipe and gets a confident answer about maximum study duration, complete
    with real citations - the worst possible failure for this system, because everything
    downstream looks correct.

    A legitimate rewrite expands the question with context from the conversation, so it keeps
    at least one of the question's own content words ("còn vừa làm vừa học thì sao?" keeps
    "vừa làm vừa học"). A hijack keeps none. When the question is pure deixis ("vậy thì
    sao?") there is nothing to preserve and the rewrite is trusted.
    """
    rewritten = rewritten.strip()
    if not rewritten:
        return False
    if len(rewritten) > REWRITE_MAX_GROWTH * max(len(question), 24):
        return False
    asked = _content_words(question)
    return not asked or bool(asked & _content_words(rewritten))


def _clean_history(history: list[dict], turns: int) -> list[dict]:
    msgs = [
        {"role": m["role"], "content": nfc(str(m.get("content", "")))[:1500]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return msgs[-2 * turns:] if turns else []


def source_view(payload: dict, score: float = 0.0, origin: str = "hybrid") -> dict:
    return {
        "doc_id": payload.get("doc_id"),
        "doc_title": payload.get("doc_title"),
        "doc_short_title": payload.get("doc_short_title"),
        "label": label(payload),
        "article": payload.get("article"),
        "article_title": payload.get("article_title"),
        "chapter": payload.get("chapter"),
        "sections": payload.get("sections"),
        "page": payload.get("page"),
        "page_end": payload.get("page_end"),
        "source_refs": payload.get("source_refs"),
        "source_file": payload.get("source_file"),
        "origin": origin,
        "score": round(score, 3),
        "text": payload.get("text"),
    }


def suggestions(result: Retrieval) -> list[dict]:
    """What the question was closest to, for a 'did you mean' line when nothing matched."""
    seen, out = set(), []
    for hit in result.fallback:
        article = hit.payload.get("article")
        if not article or article in seen:
            continue
        seen.add(article)
        out.append({"article": article, "title": hit.payload.get("article_title"),
                    "page": hit.payload.get("page")})
    return out


class ChatService:
    def __init__(self, settings: Settings, retriever: Retriever, llm: LLMClient, on_turn=None):
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.on_turn = on_turn  # called with the turn's record once it finishes (query log)

    def answer(self, message: str, history: list[dict]) -> Iterator[tuple[str, object]]:
        started = time.monotonic()
        question = nfc(message).strip()
        history = _clean_history(history, self.settings.history_turns)

        query = question
        if needs_rewrite(question, history, self.settings.rewrite_follow_ups):
            rewritten = ""
            try:
                rewritten = self.llm.complete(rewrite_messages(question, history), temperature=0)
            except Exception:  # fall back to the raw question
                log.exception("query rewrite failed")
            if rewritten and accept_rewrite(question, rewritten):
                query = rewritten
            elif rewritten:
                log.warning("rewrite rejected as off-topic: %r -> %r", question, rewritten)
            if query != question:
                yield "query", query

        result = self.retriever.relevant(query)
        record = {"question": question, "query": query, "retrieval_ms": result.took_ms,
                  "candidates": result.candidates, "model": self.settings.chat_model,
                  "expanded": (result.expansion.added if result.expansion else []),
                  "refs": [str(r) for r in result.refs]}

        if not result.hits:
            yield "token", NO_MATCH
            yield "sources", []
            done = {"found": False, "read": 0, "retrieval_ms": result.took_ms,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "suggestions": suggestions(result)}
            yield "done", done
            self._record(record | {"found": False, "latency_ms": done["latency_ms"], "sources": 0})
            return

        context, used = build_context(result.payloads, self.settings.max_context_tokens)
        origins = {id(h.payload): h.origin for h in result.hits}
        scores = {id(h.payload): h.dense_score for h in result.hits}
        views = [source_view(p, scores.get(id(p), 0.0), origins.get(id(p), "hybrid")) for p in used]
        yield "retrieved", {"sources": views, "took_ms": result.took_ms,
                            "expanded": record["expanded"]}

        answer: list[str] = []
        first_token_ms = None
        for token in self.llm.stream(answer_messages(query, context, history)):
            if first_token_ms is None:
                first_token_ms = int((time.monotonic() - started) * 1000)
            answer.append(token)
            yield "token", token

        text = "".join(answer)
        found = NOT_FOUND not in text
        bad = citations.unsupported(text, used) if found else []
        yield "sources", views if found else []
        done = {
            "found": found,
            # How many excerpts the model actually read. The UI promised the reader these
            # while the answer was being written, so when the answer turns out to be "not
            # covered", it has to say what became of them rather than leave the two
            # statements contradicting each other.
            "read": len(views),
            "retrieval_ms": result.took_ms,
            "first_token_ms": first_token_ms,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "unsupported_citations": bad,
            "sources": len(views),
        }
        yield "done", done
        self._record(record | {"found": found, "latency_ms": done["latency_ms"],
                               "first_token_ms": first_token_ms, "sources": len(views),
                               "unsupported_citations": bad})

    def _record(self, record: dict) -> None:
        if self.on_turn is None:
            return
        try:
            self.on_turn(record)
        except Exception:  # logging must never break an answer
            log.exception("query log failed")
