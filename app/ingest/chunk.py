"""Split reviewed regulation Markdown (data/processed/*.md) into retrieval chunks.

The input is the output of app.ingest.ocr after manual review:

    <!-- doc: {...} -->           document metadata (JSON)
    <!-- page: N -->              page marker
    <!-- ref: Khoản 1, Điều 2, TT17 -->   margin reference for the next paragraph/heading
    Chương N. TITLE
    Điều N. Title
    5.2 Title                     section inside an article
    PHỤ LỤC N. TITLE
    1. Title / 2.1. Title         items inside an appendix
    | a | b |                     markdown tables (never split)

Chunking rule: one chunk per article (Điều) / appendix (Phụ lục) when it fits in
CHUNK_MAX_TOKENS; otherwise consecutive sections are packed into chunks up to the limit, and a
section that is still too long is cut between paragraphs with a small overlap. Each chunk gets
a context prefix for embedding:

    [<short title> > Chương 2. ... > Điều 5. Thời gian đào tạo > 5.2 Thời gian đào tạo cụ thể]

Debug:
    python -m app.ingest.chunk --show            # one line per chunk
    python -m app.ingest.chunk --id 12           # full text + metadata of chunk 12
    python -m app.ingest.chunk --jsonl out.jsonl # dump all chunks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import get_settings

log = logging.getLogger("chunk")

DOC_RE = re.compile(r"^<!-- doc: (\{.*\}) -->$")
PAGE_RE = re.compile(r"^<!-- page: (\d+) -->$")
REF_RE = re.compile(r"^<!-- ref: (.*?)(?:\s*\|\s*raw:.*)? -->$")
CHAPTER_RE = re.compile(r"^Chương\s+(\d+|[IVXLC]+)\.\s*(.*)$")
ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
APPENDIX_RE = re.compile(r"^PHỤ LỤC\s+(\d+)\.\s*(.*)$")
SECTION_RE = re.compile(r"^(\d+\.\d+)\.?\s+(\S.*)$")
ITEM_RE = re.compile(r"^(\d+)\.\s+(\S.*)$")
SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-ZĐÀ-Ỹ\-–])")

# A chunk is never cut right after this little content (a heading, a one-line intro).
MIN_PIECE_TOKENS = 50
# An intro this small is merged into the next section rather than chunked on its own.
CARRY_TOKENS = 150

# Namespace for deterministic chunk ids (uuid5).
CHUNK_NS = uuid.UUID("6f1c2a52-6b0e-4f7a-9a51-5d1e0c3b7a10")


def estimate_tokens(text: str) -> int:
    """Rough bge-m3 (XLM-R) token count: Vietnamese syllables are mostly one token each,
    punctuation and digits add a few. Good enough to size chunks; not exact."""
    return round(1.15 * len(re.findall(r"\w+|[^\w\s]", text)))


# --- parsed structure ---


@dataclass
class Block:
    """A paragraph, list item or whole table, the smallest unit that is never cut."""

    text: str
    page: int
    is_table: bool = False

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class Unit:
    """Content under one heading: preamble, article intro, section, appendix item."""

    group: str  # key shared by everything in one article / appendix
    chapter: str | None
    chapter_title: str | None
    article: str | None  # "Điều 5" or "Phụ lục 9"
    article_title: str | None
    section: str | None  # "5.2", "3", "2.1"
    section_title: str | None
    page: int
    blocks: list[Block] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(b.tokens for b in self.blocks)


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    doc_id: str
    doc_title: str
    doc_short_title: str
    doc_number: str | None
    effective_year: int | None
    source_file: str
    chapter: str | None
    chapter_title: str | None
    article: str | None
    article_title: str | None
    sections: list[str]
    page: int
    page_end: int
    source_refs: list[str]
    token_count: int
    text: str  # body only, what the user sees as the quoted passage
    embed_text: str  # context prefix + body, what gets embedded


@dataclass
class Document:
    meta: dict
    units: list[Unit]


# --- parsing ---


def parse(markdown: str, source_name: str = "") -> Document:
    meta: dict = {}
    units: list[Unit] = []
    page = 1
    pending_refs: list[str] = []
    chapter = chapter_title = None
    group_kind: str | None = None  # "article" | "appendix" | None (preamble)
    current: Unit | None = None
    last_item = 0
    table: list[str] = []

    def new_unit(**kw) -> Unit:
        nonlocal current, pending_refs
        current = Unit(page=page, refs=pending_refs, **kw)
        pending_refs = []
        units.append(current)
        return current

    def add_block(text: str, is_table: bool = False) -> None:
        nonlocal pending_refs
        if current is None:
            new_unit(group="preamble", chapter=None, chapter_title=None, article=None,
                     article_title=None, section=None, section_title=None)
        prev = current.blocks[-1] if current.blocks else None
        if (prev and not is_table and not prev.is_table and prev.page != page
                and not prev.text.endswith((".", ":", ";", "!", "?")) and text[:1].islower()):
            prev.text = f"{prev.text} {text}"  # sentence cut by a page break
        else:
            current.blocks.append(Block(text, page, is_table))
        current.refs.extend(pending_refs)
        pending_refs = []

    def flush_table() -> None:
        if table:
            add_block("\n".join(table), is_table=True)
            table.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            flush_table()
        if not line:
            continue
        if m := DOC_RE.match(line):
            meta = json.loads(m.group(1))
            continue
        if m := PAGE_RE.match(line):
            page = int(m.group(1))
            continue
        if m := REF_RE.match(line):
            pending_refs.append(m.group(1).strip())
            continue
        if line.startswith("|"):
            table.append(line)
            continue

        if m := CHAPTER_RE.match(line):
            chapter, chapter_title = f"Chương {m.group(1)}", m.group(2).strip()
            # A chapter heading has no content of its own; its refs go to the next heading.
            continue
        if m := ARTICLE_RE.match(line):
            group_kind, last_item = "article", 0
            art = f"Điều {m.group(1)}"
            new_unit(group=art, chapter=chapter, chapter_title=chapter_title, article=art,
                     article_title=m.group(2).strip(), section=None, section_title=None)
            current.blocks.append(Block(line, page))
            continue
        if m := APPENDIX_RE.match(line):
            group_kind, last_item = "appendix", 0
            chapter = chapter_title = None  # appendices sit outside the chapters
            app = f"Phụ lục {m.group(1)}"
            new_unit(group=app, chapter=None, chapter_title=None, article=app,
                     article_title=m.group(2).strip(), section=None, section_title=None)
            current.blocks.append(Block(line, page))
            continue
        if group_kind and (m := SECTION_RE.match(line)) and len(line) < 160:
            new_unit(group=current.group, chapter=current.chapter,
                     chapter_title=current.chapter_title, article=current.article,
                     article_title=current.article_title, section=m.group(1),
                     section_title=m.group(2).strip())
            current.blocks.append(Block(line, page))
            continue
        if group_kind == "appendix" and (m := ITEM_RE.match(line)) and is_item_heading(
            line, int(m.group(1)), last_item
        ):
            last_item = int(m.group(1))
            new_unit(group=current.group, chapter=None, chapter_title=None,
                     article=current.article, article_title=current.article_title,
                     section=m.group(1), section_title=m.group(2).strip())
            current.blocks.append(Block(line, page))
            continue
        add_block(line)

    flush_table()
    if pending_refs and current:
        current.refs.extend(pending_refs)
    meta.setdefault("source_file", source_name)
    return Document(meta, units)


def is_item_heading(line: str, number: int, last: int) -> bool:
    """Appendix item titles ("2. Các điều kiện ràng buộc") versus numbered list entries."""
    return (
        number in (last + 1, 1)
        and len(line.split()) <= 20
        and not line.rstrip().endswith((".", ";", ",", ":"))
    )


# --- chunking ---


def split_long_block(block: Block, max_tokens: int) -> list[Block]:
    """Cut an over-long paragraph between sentences. Tables are never cut."""
    if block.is_table or block.tokens <= max_tokens:
        return [block]
    pieces, current = [], ""
    for sentence in SENTENCE_END.split(block.text):
        candidate = f"{current} {sentence}".strip()
        if current and estimate_tokens(candidate) > max_tokens:
            pieces.append(Block(current, block.page))
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(Block(current, block.page))
    return pieces


def bind_table_intros(blocks: list[Block]) -> list[Block]:
    """Keep the sentence that introduces a table together with the table."""
    out: list[Block] = []
    for b in blocks:
        if b.is_table and out and not out[-1].is_table and out[-1].text.endswith(":"):
            intro = out.pop()
            out.append(Block(f"{intro.text}\n\n{b.text}", intro.page, is_table=True))
        else:
            out.append(b)
    return out


def pack_blocks(blocks: list[Block], max_tokens: int, overlap: int) -> list[list[Block]]:
    blocks = [p for b in bind_table_intros(blocks) for p in split_long_block(b, max_tokens)]
    pieces: list[list[Block]] = []
    current: list[Block] = []
    for b in blocks:
        size = sum(x.tokens for x in current)
        # A piece holding only a heading or a one-line intro stays with what follows,
        # even if that pushes it over the limit (e.g. a heading followed by a large table).
        if current and size >= MIN_PIECE_TOKENS and size + b.tokens > max_tokens:
            pieces.append(current)
            tail = current[-1]
            current = [tail] if not tail.is_table and tail.tokens <= overlap else []
        current.append(b)
    if current:
        pieces.append(current)
    return pieces


def pack_units(units: list[Unit], max_tokens: int, overlap: int) -> list[tuple[list[Unit], list[Block]]]:
    """Group the units of one article/appendix into chunks; returns (units, blocks) per chunk."""
    if sum(u.tokens for u in units) <= max_tokens:
        return [(units, [b for u in units for b in u.blocks])]
    out: list[tuple[list[Unit], list[Block]]] = []
    batch: list[Unit] = []
    for unit in units:
        if unit.tokens > max_tokens:
            # A short intro before a long section (e.g. "Điều 2. Giải thích từ ngữ") is carried
            # into the section's first piece instead of becoming a chunk of its own.
            carry: list[Unit] = []
            if batch and sum(u.tokens for u in batch) <= CARRY_TOKENS:
                carry, batch = batch, []
            elif batch:
                out.append((batch, [b for u in batch for b in u.blocks]))
                batch = []
            blocks = [b for u in carry for b in u.blocks] + unit.blocks
            heading = unit.blocks[0] if unit.section else None
            for i, piece in enumerate(pack_blocks(blocks, max_tokens, overlap)):
                if i and heading and piece[0] is not heading:
                    piece = [Block(f"{heading.text} (tiếp)", piece[0].page), *piece]
                out.append(([*carry, unit] if i == 0 else [unit], piece))
            continue
        if batch and sum(u.tokens for u in batch) + unit.tokens > max_tokens:
            # Do not end a chunk on a bare heading; move it to the chunk with its content.
            held = [batch.pop()] if len(batch) > 1 and batch[-1].tokens < MIN_PIECE_TOKENS else []
            out.append((batch, [b for u in batch for b in u.blocks]))
            batch = held
        batch.append(unit)
    if batch:
        out.append((batch, [b for u in batch for b in u.blocks]))
    return out


def context_prefix(meta: dict, unit: Unit, units: list[Unit]) -> str:
    parts = [meta["short_title"]]
    if unit.chapter:
        parts.append(f"{unit.chapter}. {unit.chapter_title}".strip(". "))
    if unit.article:
        title = unit.article_title or ""
        parts.append(f"{unit.article.replace('Phụ lục', 'PHỤ LỤC')}. {title}".strip(". "))
    sections = [u for u in units if u.section]
    if len(sections) == 1:
        parts.append(f"{sections[0].section} {sections[0].section_title}")
    elif sections:
        parts.append(f"{sections[0].section}–{sections[-1].section}")
    return "[" + " > ".join(parts) + "]"


def doc_meta(doc: Document, path: Path) -> dict:
    m = dict(doc.meta)
    m.setdefault("doc_id", path.stem)
    m.setdefault("title", path.stem)
    m.setdefault("short_title", m["title"])
    m.setdefault("number", None)
    m.setdefault("effective_year", None)
    return m


def chunk_document(doc: Document, path: Path, max_tokens: int, overlap: int) -> list[Chunk]:
    meta = doc_meta(doc, path)
    groups: dict[str, list[Unit]] = {}
    for unit in doc.units:
        groups.setdefault(unit.group, []).append(unit)

    chunks: list[Chunk] = []
    for units in groups.values():
        for chunk_units, blocks in pack_units(units, max_tokens, overlap):
            if not blocks:
                continue
            first = chunk_units[0]
            body = "\n".join(f"\n{b.text}\n" if b.is_table else b.text for b in blocks).strip()
            body = re.sub(r"\n{3,}", "\n\n", body)
            prefix = context_prefix(meta, first, chunk_units)
            refs = list(dict.fromkeys(r for u in chunk_units for r in u.refs))
            index = len(chunks)
            digest = hashlib.sha256(body.encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(CHUNK_NS, f"{meta['doc_id']}|{first.article}|{index}|{digest}")),
                    chunk_index=index,
                    doc_id=meta["doc_id"],
                    doc_title=meta["title"],
                    doc_short_title=meta["short_title"],
                    doc_number=meta["number"],
                    effective_year=meta["effective_year"],
                    source_file=meta.get("source_file", path.name),
                    chapter=first.chapter,
                    chapter_title=first.chapter_title,
                    article=first.article,
                    article_title=first.article_title,
                    sections=[u.section for u in chunk_units if u.section],
                    page=min(b.page for b in blocks),
                    page_end=max(b.page for b in blocks),
                    source_refs=refs,
                    token_count=estimate_tokens(body),
                    text=body,
                    embed_text=f"{prefix}\n{body}",
                )
            )
    return chunks


def chunk_file(path: Path, max_tokens: int | None = None, overlap: int | None = None) -> list[Chunk]:
    settings = get_settings()
    doc = parse(path.read_text(encoding="utf-8"), path.name)
    return chunk_document(
        doc, path,
        max_tokens or settings.chunk_max_tokens,
        overlap if overlap is not None else settings.chunk_overlap_tokens,
    )


# --- CLI ---


def label(c: Chunk) -> str:
    where = c.article or "Mở đầu"
    if c.sections:
        where += " / " + (c.sections[0] if len(c.sections) == 1 else f"{c.sections[0]}–{c.sections[-1]}")
    return where


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk data/processed/*.md")
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--show", action="store_true", help="one line per chunk")
    parser.add_argument("--id", type=int, help="print one chunk in full")
    parser.add_argument("--jsonl", type=Path, help="write all chunks as JSON lines")
    parser.add_argument("--max-tokens", type=int, help="override CHUNK_MAX_TOKENS")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    files = args.files or sorted(Path(get_settings().processed_dir).glob("*.md"))
    chunks = [c for f in files for c in chunk_file(f, args.max_tokens)]
    limit = args.max_tokens or get_settings().chunk_max_tokens

    if args.id is not None:
        c = next((c for c in chunks if c.chunk_index == args.id), None)
        if c is None:
            sys.exit(f"no chunk {args.id}")
        meta = {k: v for k, v in asdict(c).items() if k not in ("text", "embed_text")}
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print("-" * 80)
        print(c.embed_text)
        return
    if args.jsonl:
        with args.jsonl.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        log.info("wrote %d chunks to %s", len(chunks), args.jsonl)
    if args.show or not args.jsonl:
        print(f"{'#':>3}  {'tok':>4}  {'trang':>7}  {'vị trí':<22} refs  mở đầu")
        for c in chunks:
            pages = f"{c.page}" if c.page == c.page_end else f"{c.page}-{c.page_end}"
            warn = " !" if c.token_count > limit else ""
            first_line = c.text.splitlines()[0][:60]
            print(f"{c.chunk_index:>3}  {c.token_count:>4}{warn:2} {pages:>7}  {label(c):<22} "
                  f"{len(c.source_refs):>3}  {first_line}")
        sizes = sorted(c.token_count for c in chunks)
        print(f"\n{len(chunks)} chunks, tokens min {sizes[0]} / median {sizes[len(sizes) // 2]}"
              f" / max {sizes[-1]}, over limit: {sum(s > limit for s in sizes)}")


if __name__ == "__main__":
    main()
