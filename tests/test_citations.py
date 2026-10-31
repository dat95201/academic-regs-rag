from app.rag.citations import Ref, detect, unsupported


def test_detect_article_and_section():
    assert detect("Điều 15 nói gì?") == [Ref("Điều 15")]
    assert detect("khoản 5.2 quy định sao?") == [Ref("Điều 5", "5.2")]
    assert detect("xem Điều 37.1") == [Ref("Điều 37", "37.1")]


def test_detect_appendix_and_leading_zeros():
    assert Ref("Phụ lục 7") in detect("phụ lục 07 nói về ngoại ngữ")


def test_detect_ignores_words_that_only_contain_dieu():
    assert detect("điều kiện tốt nghiệp") == []


def test_unsupported_citation_is_reported():
    payloads = [{"article": "Điều 5"}, {"article": "Phụ lục 2"}]
    answer = "Thời gian tối đa là 12 học kỳ [Quy định, Điều 5.2] và xem thêm [Quy định, Điều 99]."
    assert unsupported(answer, payloads) == ["Điều 99"]


def test_no_context_means_nothing_to_check():
    assert unsupported("bất kỳ [Quy định, Điều 3]", []) == []
