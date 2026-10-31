"""The retrieval pipeline end to end, against an in-memory Qdrant with real chunks."""

import pytest

from app.rag.retriever import fuse


class P:
    def __init__(self, id, score, payload):
        self.id, self.score, self.payload = id, score, payload


def test_fusion_keeps_each_branch_score_and_rewards_agreement():
    dense = [P("a", 0.9, {}), P("b", 0.8, {})]
    sparse = [P("b", 4.0, {}), P("c", 3.0, {})]
    hits = {h.id: h for h in fuse(dense, sparse, 1.0, 1.0)}
    assert hits["b"].dense_score == 0.8 and hits["b"].sparse_score == 4.0
    # "b" is in both branches, so it outranks the top of either one alone
    assert max(hits.values(), key=lambda h: h.score).id == "b"


def test_sparse_weight_zero_gives_pure_dense_order():
    out = fuse([P("a", 0.9, {}), P("b", 0.1, {})], [P("b", 9.0, {})], 1.0, 0.0)
    assert [h.id for h in out] == ["a", "b"]


def test_search_finds_the_article_about_maximum_duration(retriever):
    hits = retriever.search("thời gian đào tạo tối đa của hệ chính quy")
    assert hits
    assert "Điều 5" in {h.payload.get("article") for h in hits[:5]}


def test_explicit_article_number_is_fetched_even_if_vectors_disagree(retriever):
    result = retriever.retrieve("Điều 28 quy định gì")
    assert [str(r) for r in result.refs] == ["Điều 28"]
    assert "Điều 28" in {h.payload.get("article") for h in result.hits}


def test_relevant_applies_the_gate(retriever, settings):
    settings.min_relevance_score = 0.99
    try:
        result = retriever.relevant("công thức nấu phở bò gia truyền")
        assert result.gated and result.hits == []
        assert result.fallback  # still offers the closest articles
    finally:
        settings.min_relevance_score = 0.05


def test_number_of_sources_follows_relevance_not_a_quota(retriever, settings):
    """Regression: the gate used to be all-or-nothing, so every answer carried exactly
    TOP_K_FINAL (+ neighbours) 'sources' however many were actually relevant."""
    tight = retriever.relevant("thời hạn yêu cầu phúc tra bài thi")
    settings.relative_cutoff = 0.0  # back to the old behaviour: keep the whole top-k
    try:
        loose = retriever.relevant("thời hạn yêu cầu phúc tra bài thi")
    finally:
        settings.relative_cutoff = 0.85
    assert len(tight.hits) <= len(loose.hits)
    # TOP_K_FINAL is now a ceiling on the whole set: neighbours fill the slots the relevance
    # cutoff left empty, instead of being added on top of a full quota.
    assert len(loose.hits) == settings.top_k_final


def test_a_chunk_far_behind_the_best_one_is_dropped(retriever, settings):
    result = retriever.relevant("điều kiện được công nhận tốt nghiệp")
    scored = [h.dense_score for h in result.hits if h.origin == "hybrid" and h.dense_score > 0]
    if len(scored) > 1:
        assert min(scored) >= max(scored) * settings.relative_cutoff - 1e-9


def test_named_article_survives_the_cutoff_whatever_its_score(retriever):
    result = retriever.relevant("Điều 28 quy định gì")
    assert "Điều 28" in {h.payload.get("article") for h in result.hits}


def test_neighbours_only_fill_the_room_left_by_top_k(retriever, settings):
    result = retriever.relevant("điều kiện cảnh báo học tập và buộc thôi học")
    assert len(result.hits) <= settings.top_k_final


def test_relevant_widens_with_neighbours(retriever, settings):
    result = retriever.relevant("điều kiện cảnh báo học tập và buộc thôi học")
    assert result.hits
    assert len(result.hits) <= settings.top_k_final + settings.context_neighbours
    assert all(h.origin in ("hybrid", "article", "context") for h in result.hits)


def test_context_expansion_can_be_disabled(retriever, settings):
    settings.context_expansion = False
    try:
        result = retriever.relevant("đăng ký học phần tối đa")
        assert all(h.origin != "context" for h in result.hits)
        assert len(result.hits) <= settings.top_k_final
    finally:
        settings.context_expansion = True


def test_glossary_expansion_is_reported(retriever):
    result = retriever.retrieve("GPA thấp thì sao")
    assert "điểm trung bình tích luỹ" in (result.expansion.added if result.expansion else [])


def test_hit_view_is_json_friendly(retriever):
    view = retriever.search("học phí")[0].view()
    assert {"id", "score", "cosine", "bm25", "origin", "article"} <= set(view)


@pytest.mark.parametrize("question", [
    "khi nào sinh viên bị cảnh báo học tập",
    "mỗi học kỳ đăng ký tối thiểu bao nhiêu tín chỉ",
    "điều kiện được công nhận tốt nghiệp",
    "thời hạn yêu cầu phúc tra bài thi",
])
def test_core_questions_return_something(retriever, question):
    assert retriever.relevant(question).hits
