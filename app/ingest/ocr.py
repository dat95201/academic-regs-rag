"""OCR scanned PDFs in data/raw into reviewable Markdown in data/processed.

Run once per document; the result is cached by the PDF's SHA-256 so re-running is a no-op
until the source changes. The Markdown output is meant to be reviewed and corrected by hand
(tables, margin references) and then becomes the input of the ingest pipeline.

    python -m app.ingest.ocr                 # every PDF in data/raw without a text layer
    python -m app.ingest.ocr --force         # ignore the cache
    python -m app.ingest.ocr --pages 1-4     # a page range, for tuning (output: *.pages-1-4.md)

Requires the `tesseract` binary with the `vie` language pack.

Output format (one file per document, data/processed/<doc_id>.md):

    <!-- doc: {"doc_id": ..., "source_file": ..., "sha256": ..., "ocr": "tesseract-vie"} -->
    <!-- page: 1 -->
    ...body text...
    <!-- ref: Khoản 1, Điều 2, TT17 | raw: "Khoản l, / Điều 2, / TT17" -->

`ref` comments carry the right-margin source annotations of the consolidated regulation
(which original document/article each paragraph comes from). They are placed just before the
body line they sit next to on the page. `?` in a ref marks a part the OCR could not read
reliably; check those by eye.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
import pytesseract
from PIL import Image

log = logging.getLogger("ocr")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
CACHE_FILE = OUT_DIR / ".ocr_cache.json"

BODY_DPI = 300
MARGIN_DPI = 500
# The margin annotations start at ~88.8% of the page width; body lines end at ~88.3%.
MARGIN_X = 0.884
# Light grey pixels are mostly bleed-through from the back of the sheet.
BLEED_THRESHOLD = 165
MIN_TEXT_CHARS_PER_PAGE = 20

SOURCES = ("TT08", "TT17", "HVu_chung", "HVu_ĐH")
HEADING_OR_ITEM = re.compile(
    r"^(Chương\s+(\d+|[IVXLC]+)\.|Điều\s+\d+|PHỤ LỤC\s*\d+|\d+(\.\d+)*\.?\s|[-–•o]\s|[a-zđ]\)\s)"
)
# Headings are one line; never append the following line to them.
HEADING = re.compile(r"^(Chương\s+(\d+|[IVXLC]+)\.|Điều\s+\d+\.|\d+\.\d+\.?\s|PHỤ LỤC\s*\d+)")
REF_START = re.compile(r"^(K\s?ho|Kho|Đi|Di|Ph)", re.IGNORECASE)


@dataclass
class Line:
    y: float  # top, in fractions of page height
    text: str
    bottom: float = 0.0


# --- image helpers ---


def render(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.w, pix.h), pix.samples)


def suppress_bleed(img: Image.Image) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    a = np.where(a > BLEED_THRESHOLD, 255, a)
    a = np.clip((a - 40) * 1.3, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def tsv_lines(img: Image.Image, psm: int, keep) -> list[tuple[tuple, list[dict]]]:
    """Tesseract words grouped by (block, paragraph, line), filtered per block by `keep`."""
    d = pytesseract.image_to_data(
        img, lang="vie", config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    blocks: dict[int, list[dict]] = {}
    for i, word in enumerate(d["text"]):
        if not word.strip():
            continue
        blocks.setdefault(d["block_num"][i], []).append(
            {
                "key": (d["block_num"][i], d["par_num"][i], d["line_num"][i]),
                "left": d["left"][i],
                "top": d["top"][i],
                "height": d["height"][i],
                "text": word,
            }
        )
    lines: dict[tuple, list[dict]] = {}
    for words in blocks.values():
        if not keep(words):
            continue
        for w in words:
            lines.setdefault(w["key"], []).append(w)
    return sorted(lines.items(), key=lambda kv: kv[0])


# --- body ---


def clean_line(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"^[-–_]\s*[-_]\s*", "- ", text)  # "-_ ", "- _ ", "- - " bullets
    text = re.sub(r"^o_\s*", "o ", text)  # sub-bullets
    text = re.sub(r"^[-–—\s]+(?=PHỤ LỤC)", "", text)
    text = re.sub(r"^PHỤ LỤC\s*(\d+)", r"PHỤ LỤC \1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_junk(text: str) -> bool:
    letters = sum(c.isalnum() for c in text)
    return letters < 3 or letters / max(len(text), 1) < 0.5


def body_lines(page: pymupdf.Page, masked: list[tuple[float, float]]) -> list[Line]:
    img = suppress_bleed(render(page, BODY_DPI))
    width, height = img.size
    # Blank the margin column where annotations sit, so they do not leak into body lines.
    # Elsewhere the column is kept: some body lines (e.g. the page-1 preamble) run into it.
    a = np.asarray(img).copy()
    for top, bottom in masked:
        a[int(top * height):int(bottom * height) + 1, int(MARGIN_X * width):] = 255
    img = Image.fromarray(a)

    def not_margin_block(words: list[dict]) -> bool:
        # A block whose every word sits in the margin is an annotation, not body text.
        return any(w["left"] < MARGIN_X * width for w in words)

    paragraphs: list[tuple[tuple, list[Line]]] = []
    for key, words in tsv_lines(img, 3, not_margin_block):
        text = clean_line(" ".join(w["text"] for w in sorted(words, key=lambda w: w["left"])))
        if not text or is_junk(text):
            continue
        line = Line(min(w["top"] for w in words) / height, text)
        par = key[:2]
        if paragraphs and paragraphs[-1][0] == par:
            paragraphs[-1][1].append(line)
        else:
            paragraphs.append((par, [line]))

    out: list[Line] = []
    for _, lines in paragraphs:
        current: Line | None = None
        for line in lines:
            if current is None or HEADING_OR_ITEM.match(line.text) or HEADING.match(current.text):
                if current:
                    out.append(current)
                current = Line(line.y, line.text)
            else:
                current.text += " " + line.text
        if current:
            out.append(current)

    if out and re.fullmatch(r"\d{1,3}", out[0].text):  # page number at the top
        out = out[1:]
    return out


# --- margin references ---


def normalize_source(raw: str) -> str:
    s = raw.replace(" ", "")
    plain = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    if "chung" in plain.lower():
        return "HVu_chung"
    if re.search(r"Đ[HI]|DH|ĐH", s):
        return "HVu_ĐH"
    if re.search(r"08|108|098", s):
        return "TT08"
    if re.search(r"17", s):
        return "TT17"
    if re.search(r"T", s):
        return "TT??"
    return "?"


def normalize_number(raw: str) -> str:
    table = str.maketrans({"Ì": "1", "Í": "1", "l": "1", "I": "1", "|": "1", "O": "0", "o": "0"})
    num = raw.translate(table).strip(" ,.")
    return num if re.fullmatch(r"\d+(\.\d+)?", num) else "?"


def parse_ref(parts: list[str]) -> str:
    """Best-effort 'Khoản X, Điều Y, SRC' from the OCR lines of one margin note."""
    joined = " ".join(parts)
    pieces = []
    m = re.search(r"K\s?ho[aảáàạ]n\s+([^\s,]+)", joined, re.IGNORECASE)
    if m:
        pieces.append(f"Khoản {normalize_number(m.group(1))}")
    m = re.search(r"(?:Đi|Di)[eêềểệẻ]\w*\s+([^\s,]+)", joined)
    if m:
        pieces.append(f"Điều {normalize_number(m.group(1))}")
    m = re.search(r"Ph[uụ]\s*l[uụ]c\s+([^\s,]+)", joined, re.IGNORECASE)
    if m:
        pieces.append(f"Phụ lục {normalize_number(m.group(1))}")
    source = [p for p in parts if not REF_START.match(p)]
    pieces.append(normalize_source(source[-1]) if source else "?")
    return ", ".join(pieces)


def margin_refs(page: pymupdf.Page) -> tuple[list[Line], list[tuple[float, float]]]:
    img = render(page, MARGIN_DPI)
    width, height = img.size
    strip = img.crop((int(MARGIN_X * width), 0, width, height)).point(
        lambda v: 255 if v > 170 else v
    )
    raw_lines = []
    for _, words in tsv_lines(strip, 4, lambda _: True):
        text = clean_line(" ".join(w["text"] for w in sorted(words, key=lambda w: w["left"])))
        if text and sum(c.isalnum() for c in text) >= 2:
            raw_lines.append(Line(min(w["top"] for w in words) / height, text,
                                  max(w["top"] + w["height"] for w in words) / height))

    # Group lines into notes like "Khoản 1, / Điều 2, / TT17". A note ends at its source line,
    # so a "Khoản/Điều/Phụ lục" line after a non-heading line starts the next note.
    notes: list[list[Line]] = []
    for line in raw_lines:
        starts = REF_START.match(line.text) is not None
        gap = bool(notes) and line.y - notes[-1][-1].y > 0.02
        after_source = bool(notes) and REF_START.match(notes[-1][-1].text) is None
        if not notes or gap or (starts and after_source):
            notes.append([line])
        else:
            notes[-1].append(line)
    refs, masked = [], []
    for note in notes:
        if not REF_START.match(note[0].text) or len(note) < 2:
            continue  # stray body fragments that spilled into the margin
        parts = [n.text for n in note]
        refs.append(Line(note[0].y, f'<!-- ref: {parse_ref(parts)} | raw: "{" / ".join(parts)}" -->'))
        masked.append((note[0].y - 0.004, note[-1].bottom + 0.004))
    return refs, masked


# --- document ---


def merge(body: list[Line], refs: list[Line]) -> list[str]:
    out: list[str] = []
    pending = sorted(refs, key=lambda r: r.y)
    for line in body:
        while pending and pending[0].y <= line.y + 0.01:
            out.append(pending.pop(0).text)
        out.append(line.text)
    out.extend(r.text for r in pending)
    return out


def has_text_layer(doc: pymupdf.Document) -> bool:
    chars = sum(len(p.get_text().strip()) for p in doc)
    return chars >= MIN_TEXT_CHARS_PER_PAGE * doc.page_count


def doc_id_for(path: Path) -> str:
    s = unicodedata.normalize("NFD", path.stem).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def page_range(spec: str | None, count: int) -> range:
    if not spec:
        return range(count)
    start, _, end = spec.partition("-")
    return range(int(start) - 1, int(end or start))


def ocr_pdf(path: Path, pages: range) -> str:
    doc = pymupdf.open(path)
    meta = {"doc_id": doc_id_for(path), "source_file": path.name, "sha256": sha256(path),
            "ocr": "tesseract-vie"}
    out = [f"<!-- doc: {json.dumps(meta, ensure_ascii=False)} -->", ""]
    for i in pages:
        log.info("%s: page %d/%d", path.name, i + 1, doc.page_count)
        page = doc[i]
        out.append(f"<!-- page: {i + 1} -->")
        refs, masked = margin_refs(page)
        out.extend(merge(body_lines(page, masked), refs))
        out.append("")
    return "\n".join(out)


def report(markdown: str) -> list[str]:
    """Structural problems worth a manual look: numbering gaps and unreadable refs."""
    problems = []
    for label, pattern in (("Chương", r"^Chương (\d+)\."), ("Điều", r"^Điều (\d+)\."),
                           ("PHỤ LỤC", r"^PHỤ LỤC (\d+)\.")):
        nums = [int(n) for n in re.findall(pattern, markdown, re.M)]
        expected = list(range(1, max(nums, default=0) + 1))
        missing = sorted(set(expected) - set(nums))
        out_of_order = [b for a, b in zip(nums, nums[1:], strict=False) if b != a + 1]
        if missing or out_of_order:
            problems.append(f"{label}: missing {missing}, out of order after {out_of_order}")
    unreadable = len(re.findall(r"<!-- ref: [^|]*\?", markdown))
    if unreadable:
        problems.append(f"{unreadable} margin refs contain '?' (grep 'ref: .*?')")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--pages", help="page range like 1-4 (tuning only, not cached)")
    parser.add_argument("--check", action="store_true",
                        help="only report structural problems in data/processed/*.md")
    parser.add_argument("files", nargs="*", type=Path, help="PDFs (default: data/raw/*.pdf)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.check:
        for md in sorted(OUT_DIR.glob("*.md")):
            for problem in report(md.read_text(encoding="utf-8")) or ["OK"]:
                print(f"{md.name}: {problem}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    for path in args.files or sorted(RAW_DIR.glob("*.pdf")):
        doc = pymupdf.open(path)
        if has_text_layer(doc):
            log.info("%s has a text layer, skipping OCR", path.name)
            continue
        digest = sha256(path)
        key = unicodedata.normalize("NFC", path.name)  # macOS file names are NFD
        target = OUT_DIR / f"{doc_id_for(path)}.md"
        if args.pages:
            target = target.with_suffix(f".pages-{args.pages}.md")
        elif not args.force and cache.get(key) == digest and target.exists():
            log.info("%s unchanged, keeping %s", path.name, target)
            continue
        if target.exists() and not args.pages and not args.force:
            # Never overwrite a file that may hold manual corrections.
            log.warning("%s exists and the PDF changed; writing %s.new for diffing",
                        target, target.name)
            target = target.with_suffix(".md.new")
        markdown = ocr_pdf(path, page_range(args.pages, doc.page_count))
        target.write_text(markdown, encoding="utf-8")
        log.info("wrote %s", target)
        for problem in report(markdown):
            log.warning("%s: %s", target.name, problem)
        if not args.pages:
            cache[key] = digest
            CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
