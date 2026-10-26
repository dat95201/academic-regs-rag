"""One chat turn: rewrite follow-up -> retrieve -> answer from excerpts, as a stream of events."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from app.config import Settings
from app.rag.embeddings import nfc
from app.rag.llm import LLMClient
from app.rag.prompt import NOT_FOUND, answer_messages, build_context, label, rewrite_messages
from app.rag.retriever import Retriever

log = logging.getLogger("chat")

NO_MATCH = (
    "Không tìm thấy nội dung liên quan trong tài liệu hiện có. "
    "Bạn thử hỏi cụ thể hơn (ví dụ nêu rõ học phần, học kỳ, loại chương trình)."
)


def _clean_history(history: list[dict], turns: int) -> list[dict]:
    msgs = [
        {"role": m["role"], "content": nfc(str(m.get("content", "")))[:1500]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return msgs[-2 * turns:] if turns else []


def source_view(payload: dict, score: float) -> dict:
    return {
        "doc_title": payload.get("doc_title"),
        "doc_short_title": payload.get("doc_short_title"),
        "label": label(payload),
        "article": payload.get("article"),
        "article_title": payload.get("article_title"),
        "sections": payload.get("sections"),
        "page": payload.get("page"),
        "page_end": payload.get("page_end"),
        "source_refs": payload.get("source_refs"),
        "source_file": payload.get("source_file"),
        "score": round(score, 3),
        "text": payload.get("text"),
    }


class ChatService:
    def __init__(self, settings: Settings, retriever: Retriever, llm: LLMClient):
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    def answer(self, message: str, history: list[dict]) -> Iterator[tuple[str, object]]:
        """Yields (event, data): ("query", str) ("token", str) ("sources", list) ("done", {})."""
        question = nfc(message).strip()
        history = _clean_history(history, self.settings.history_turns)

        query = question
        if history:
            try:
                query = self.llm.complete(rewrite_messages(question, history), temperature=0) or question
            except Exception:  # noqa: BLE001 - fall back to the raw question
                log.exception("query rewrite failed")
            if query != question:
                yield "query", query

        hits = self.retriever.relevant(query)
        if not hits:
            yield "token", NO_MATCH
            yield "sources", []
            yield "done", {"found": False}
            return

        context, used = build_context([h.payload for h in hits], self.settings.max_context_tokens)
        scores = {id(h.payload): h.dense_score for h in hits}
        answer = []
        for token in self.llm.stream(answer_messages(query, context, history)):
            answer.append(token)
            yield "token", token
        found = NOT_FOUND not in "".join(answer)
        yield "sources", [source_view(p, scores.get(id(p), 0.0)) for p in used] if found else []
        yield "done", {"found": found}
