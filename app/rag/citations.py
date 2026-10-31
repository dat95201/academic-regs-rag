"""Article references: find them in a question, and check the ones an answer claims.

Two jobs, same grammar:

* `detect(question)` - "Điều 15 nói gì?", "khoản 5.2", "phụ lục 7" -> an exact lookup in
  Qdrant, which beats any vector search at finding a numbered article.
* `unsupported(answer, payloads)` - the system prompt orders a citation per claim, like
  "[Quy định học vụ ĐH (hợp nhất 2022), Điều 37.1]". A small model sometimes cites an article
  that was never in the context. Those are listed so the UI can warn instead of the reader
  trusting a number that came from nowhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "Điều 15", "điều 15.2", "Đ.15"; "Phụ lục 7", "PHỤ LỤC 07"; "khoản 5.2", "mục 37.1".
ARTICLE_RE = re.compile(r"(?<!\w)Đi[eề]u\s*0*(\d+)(?:\s*[.,]\s*0*(\d+))?", re.IGNORECASE)
APPENDIX_RE = re.compile(r"(?<!\w)Ph[uụ]\s*l[uụ]c\s*0*(\d+)", re.IGNORECASE)
SECTION_RE = re.compile(r"(?<!\w)(?:kho[aả]n|m[uụ]c)\s*0*(\d+)\.(\d+)", re.IGNORECASE)
CITATION_RE = re.compile(r"\[([^\[\]]{3,160})\]")


@dataclass(frozen=True)
class Ref:
    """'Điều 37' / 'Phụ lục 7', with the section number when one was given."""

    article: str
    section: str | None = None

    def __str__(self) -> str:
        return f"{self.article}.{self.section}" if self.section else self.article


def detect(text: str) -> list[Ref]:
    """Article/appendix numbers named in a question, in the order they appear."""
    found: list[Ref] = []

    def add(ref: Ref) -> None:
        if ref not in found:
            found.append(ref)

    for m in ARTICLE_RE.finditer(text):
        add(Ref(f"Điều {int(m.group(1))}", m.group(2) and f"{int(m.group(1))}.{m.group(2)}"))
    for m in APPENDIX_RE.finditer(text):
        add(Ref(f"Phụ lục {int(m.group(1))}"))
    for m in SECTION_RE.finditer(text):
        add(Ref(f"Điều {int(m.group(1))}", f"{int(m.group(1))}.{m.group(2)}"))
    return found


def covered(payloads: list[dict]) -> set[str]:
    """Article/appendix names present in the excerpts given to the model."""
    return {p["article"] for p in payloads if p.get("article")}


def unsupported(answer: str, payloads: list[dict]) -> list[str]:
    """Citations in the answer naming an article that was not in the context."""
    allowed = covered(payloads)
    if not allowed:
        return []
    bad: list[str] = []
    for m in CITATION_RE.finditer(answer):
        for ref in detect(m.group(1)):
            if ref.article not in allowed and ref.article not in bad:
                bad.append(ref.article)
    return bad
