from app.rag.glossary import expand, has_diacritics, strip_accents


def test_acronym_is_expanded_to_the_regulation_wording():
    out = expand("GPA bao nhiêu thì bị cảnh cáo?")
    assert "điểm trung bình tích luỹ" in out.text
    assert "cảnh báo học tập" in out.text
    assert out.changed


def test_slang_is_mapped_to_the_official_phrase():
    assert "buộc thôi học" in expand("rớt nhiều môn có bị đuổi học không?").text


def test_two_letter_acronyms_need_upper_case():
    assert "tín chỉ" in expand("đăng ký bao nhiêu TC một kỳ?").text
    # "tc" inside ordinary lower-case text is not an abbreviation
    assert expand("mình thắc mắc chuyện này").added == []


def test_query_without_diacritics_still_finds_the_terms():
    out = expand("hoc phan khong dat thi sao")
    assert "học phần" in out.text


def test_phrase_already_present_is_not_appended_twice():
    out = expand("điều kiện buộc thôi học là gì?")
    assert out.text.count("buộc thôi học") == 1


def test_expansion_can_be_switched_off():
    assert expand("GPA", enabled=False).text == "GPA"


def test_strip_accents():
    assert strip_accents("Điều 15 học phần") == "dieu 15 hoc phan"
    assert has_diacritics("học") and not has_diacritics("hoc")
