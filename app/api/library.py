"""Reading the regulation as a document rather than as chunks.

The chat answers questions; sometimes a reader just wants to open Điều 15 and read it, or
jump to the page in the original PDF. Both come from the same reviewed Markdown the ingest
pipeline uses, parsed with `app.ingest.chunk.parse`, so the table of contents can never drift
away from what was indexed.

Parsed documents are cached and re-read when the file's mtime changes.
"""

from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.ingest.chunk import Document, doc_meta, parse

_lock = threading.Lock()
_cache: dict[Path, tuple[float, Document, dict]] = {}


def _load(path: Path) -> tuple[Document, dict]:
    mtime = path.stat().st_mtime
    with _lock:
        cached = _cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
    doc = parse(path.read_text(encoding="utf-8"), path.name)
    meta = doc_meta(doc, path)
    with _lock:
        _cache[path] = (mtime, doc, meta)
    return doc, meta


def _body(unit) -> list:
    """The unit's blocks without the heading line it starts with.

    `parse` keeps the heading ("Điều 5. Thời gian đào tạo", "5.2 Thời gian đào tạo cụ thể") as
    the first block so a chunk reads as a whole passage. A reader is shown the heading from
    `article`/`section` already, so repeating it as the first paragraph just looks like a bug.
    """
    blocks = unit.blocks
    if not blocks:
        return blocks
    head = blocks[0].text.strip()
    for marker in (unit.section, unit.article):
        if marker and head.startswith(marker):
            return blocks[1:]
    return blocks


@dataclass
class Library:
    processed_dir: Path
    raw_dir: Path

    def files(self) -> list[Path]:
        return sorted(p for p in self.processed_dir.glob("*.md") if ".pages-" not in p.name)

    def documents(self) -> list[tuple[Document, dict, Path]]:
        return [(*_load(p), p) for p in self.files()]

    def source_pdf(self, doc_id: str) -> Path | None:
        """The original file a document was built from, if it is still in data/raw.

        Matching is done on NFC-normalised names against the real directory listing, not by
        building a path from the metadata. macOS stores file names decomposed (NFD) while the
        metadata written during OCR is composed (NFC); the two are the same string to macOS
        but two different byte sequences to the Linux filesystem inside the container, so a
        direct `raw_dir / name` lookup works on the host and fails in Docker.
        """
        for _, meta, _ in self.documents():
            if meta["doc_id"] != doc_id:
                continue
            name = meta.get("source_file")
            if not name or not self.raw_dir.is_dir():
                return None
            wanted = unicodedata.normalize("NFC", name)
            for candidate in self.raw_dir.iterdir():
                if not candidate.is_file() or candidate.name.startswith("."):
                    continue
                if unicodedata.normalize("NFC", candidate.name) == wanted:
                    # Never follow a name out of data/raw, whatever the metadata says.
                    resolved = candidate.resolve()
                    if self.raw_dir.resolve() in resolved.parents:
                        return resolved
        return None

    def toc(self) -> list[dict]:
        """Chapters -> articles, plus the appendices, for the browse page."""
        out = []
        for doc, meta, _ in self.documents():
            chapters: list[dict] = []
            appendices: list[dict] = []
            seen: set[str] = set()
            for unit in doc.units:
                if not unit.article or unit.group in seen:
                    continue
                seen.add(unit.group)
                entry = {"article": unit.article, "title": unit.article_title, "page": unit.page}
                if unit.article.startswith("Phụ lục"):
                    appendices.append(entry)
                    continue
                if not chapters or chapters[-1]["chapter"] != unit.chapter:
                    chapters.append({"chapter": unit.chapter, "title": unit.chapter_title,
                                     "articles": []})
                chapters[-1]["articles"].append(entry)
            out.append({
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "short_title": meta["short_title"],
                "number": meta.get("number"),
                "effective_year": meta.get("effective_year"),
                "has_pdf": self.source_pdf(meta["doc_id"]) is not None,
                "chapters": chapters,
                "appendices": appendices,
            })
        return out

    def article(self, doc_id: str, article: str) -> dict | None:
        """Full text of one article / appendix, section by section."""
        for doc, meta, _ in self.documents():
            if meta["doc_id"] != doc_id:
                continue
            units = [u for u in doc.units if u.article == article]
            if not units:
                return None
            sections = [
                {
                    "section": u.section,
                    "title": u.section_title,
                    "page": u.page,
                    "refs": u.refs,
                    "text": "\n".join(b.text for b in _body(u)),
                }
                for u in units
            ]
            pages = [b.page for u in units for b in u.blocks] or [units[0].page]
            return {
                "doc_id": doc_id,
                "doc_title": meta["title"],
                "doc_short_title": meta["short_title"],
                "article": article,
                "title": units[0].article_title,
                "chapter": units[0].chapter,
                "chapter_title": units[0].chapter_title,
                "page": min(pages),
                "page_end": max(pages),
                "sections": sections,
            }
        return None
