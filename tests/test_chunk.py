"""Invariants of the chunker, checked against the real reviewed document."""

from app.ingest.chunk import Block, estimate_tokens, is_item_heading, parse, split_long_block


def test_every_chunk_has_text_and_provenance(chunks):
    assert chunks
    for c in chunks:
        assert c.text.strip()
        assert c.doc_id and c.doc_title and c.source_file
        assert c.page <= c.page_end
        assert c.embed_text.startswith("[") and c.text in c.embed_text


def test_chunk_ids_are_deterministic(markdown):
    from app.ingest.chunk import chunk_file

    a = [c.chunk_id for c in chunk_file(markdown, 800, 100)]
    b = [c.chunk_id for c in chunk_file(markdown, 800, 100)]
    assert a == b and len(set(a)) == len(a)


def test_only_tables_are_allowed_past_the_token_limit(chunks):
    over = [c for c in chunks if c.token_count > 800 * 1.35]
    assert all("|" in c.text for c in over), [c.article for c in over]


def test_articles_and_appendices_are_all_represented(chunks, markdown):
    import re

    text = markdown.read_text(encoding="utf-8")
    expected = set(re.findall(r"^(Điều \d+)\.", text, re.M)) | set(
        f"Phụ lục {n}" for n in re.findall(r"^PHỤ LỤC (\d+)\.", text, re.M))
    assert expected - {c.article for c in chunks} == set()


def test_sections_stay_inside_their_article(chunks):
    for c in chunks:
        for section in c.sections:
            if c.article and c.article.startswith("Điều") and "." in section:
                assert section.split(".")[0] == c.article.split()[-1]


def test_parse_reads_the_document_header(markdown):
    doc = parse(markdown.read_text(encoding="utf-8"), markdown.name)
    assert doc.meta.get("doc_id") and doc.meta.get("short_title")
    assert any(u.article and u.article.startswith("Điều") for u in doc.units)


def test_tables_are_never_split():
    table = Block("| a | b |\n| - | - |\n" + "| x | y |\n" * 500, 1, is_table=True)
    assert split_long_block(table, 50) == [table]


def test_long_paragraph_is_cut_between_sentences():
    text = " ".join(f"Câu số {i} dài vừa đủ để tách ra." for i in range(60))
    pieces = split_long_block(Block(text, 1), 100)
    assert len(pieces) > 1 and all(estimate_tokens(p.text) <= 130 for p in pieces)


def test_item_heading_detection():
    assert is_item_heading("2. Các điều kiện ràng buộc", 2, 1)
    assert not is_item_heading("2. Sinh viên phải nộp hồ sơ trước tuần thứ tư.", 2, 1)
