from dataclasses import dataclass, field

from app.rag.citations import Ref
from app.rag.rerank import coverage, heuristic, terms, weights


@dataclass
class FakeHit:
    score: float
    payload: dict
    rerank_score: float | None = field(default=None)


def hit(score, article, text, title=""):
    return FakeHit(score, {"article": article, "text": text, "article_title": title,
                           "sections": []})


def test_rare_terms_weigh_more_than_common_ones():
    w = weights(terms("sinh viên phúc tra"), ["sinh viên a", "sinh viên b", "sinh viên phúc tra"])
    assert w["phúc tra"] > w["sinh viên"]


def test_coverage_is_a_share_of_the_query_weight():
    query = terms("phúc tra bài thi")
    w = weights(query, ["phúc tra bài thi", "điều khác"])
    assert coverage(query, w, "quy trình phúc tra bài thi") == 1.0
    assert coverage(query, w, "không liên quan") == 0.0


def test_lexical_overlap_lifts_a_lower_ranked_chunk():
    hits = [hit(1.0, "Điều 1", "quy định chung không liên quan"),
            hit(0.9, "Điều 28", "sinh viên yêu cầu chấm phúc tra bài thi tự luận")]
    assert heuristic("phúc tra bài thi", hits)[0].payload["article"] == "Điều 28"


def test_named_article_wins():
    hits = [hit(1.0, "Điều 1", "văn bản này quy định chung"),
            hit(0.2, "Điều 15", "đăng ký học phần")]
    out = heuristic("Điều 15 nói gì", hits, refs=[Ref("Điều 15")])
    assert out[0].payload["article"] == "Điều 15"


def test_diversity_prefers_a_new_article_over_a_third_chunk_of_the_same_one():
    hits = [hit(1.0, "Điều 5", "thời gian đào tạo tối đa"),
            hit(0.99, "Điều 5", "thời gian đào tạo chuẩn"),
            hit(0.98, "Điều 5", "thời gian đào tạo kế hoạch"),
            hit(0.9, "Điều 6", "tổ chức năm học thời gian")]
    articles = [h.payload["article"] for h in heuristic("thời gian đào tạo", hits, top_k=3)]
    assert "Điều 6" in articles


def test_top_k_is_respected_and_scores_are_attached():
    hits = [hit(1.0, "Điều 1", "a"), hit(0.5, "Điều 2", "b"), hit(0.1, "Điều 3", "c")]
    out = heuristic("a", hits, top_k=2)
    assert len(out) == 2 and all(h.rerank_score is not None for h in out)


def test_empty_input():
    assert heuristic("gì cũng được", []) == []
