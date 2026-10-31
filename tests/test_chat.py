"""The chat turn: event order, the no-match path, and the citation guard."""

from app.rag.chat import ChatService, accept_rewrite, needs_rewrite


def events(service, message, history=None):
    return list(service.answer(message, history or []))


def test_event_order_and_payloads(settings, retriever):
    from tests.conftest import FakeLLM

    llm = FakeLLM(chunks=["Thời gian tối đa là 12 học kỳ ", "[Quy định, Điều 5]"])
    service = ChatService(settings, retriever, llm)
    names = [e for e, _ in events(service, "thời gian đào tạo tối đa là bao nhiêu")]
    assert names[0] == "retrieved"
    assert names[-2:] == ["sources", "done"]
    assert "token" in names


def test_retrieved_arrives_before_the_first_token(settings, retriever):
    from tests.conftest import FakeLLM

    service = ChatService(settings, retriever, FakeLLM(chunks=["a", "b"]))
    names = [e for e, _ in events(service, "điều kiện tốt nghiệp")]
    assert names.index("retrieved") < names.index("token")


def test_no_match_is_reported_with_suggestions(settings, retriever):
    from tests.conftest import FakeLLM

    settings.min_relevance_score = 0.99
    try:
        service = ChatService(settings, retriever, FakeLLM())
        out = dict(events(service, "cách làm bánh mì thịt nướng"))
        assert out["sources"] == []
        assert out["done"]["found"] is False
        assert "suggestions" in out["done"]
    finally:
        settings.min_relevance_score = 0.05


def test_unsupported_citation_is_flagged(settings, retriever):
    from tests.conftest import FakeLLM

    service = ChatService(settings, retriever, FakeLLM(chunks=["Theo [Quy định, Điều 999] thì..."]))
    done = dict(events(service, "thời gian đào tạo tối đa"))["done"]
    assert done["unsupported_citations"] == ["Điều 999"]


def test_turn_is_logged(settings, retriever):
    from tests.conftest import FakeLLM

    records = []
    service = ChatService(settings, retriever, FakeLLM(chunks=["ok"]), on_turn=records.append)
    events(service, "học phí nộp khi nào")
    assert records and records[0]["question"] == "học phí nộp khi nào"
    assert records[0]["latency_ms"] >= 0


def test_a_broken_logger_never_breaks_the_answer(settings, retriever):
    from tests.conftest import FakeLLM

    def boom(_):
        raise RuntimeError("disk full")

    service = ChatService(settings, retriever, FakeLLM(chunks=["ok"]), on_turn=boom)
    assert dict(events(service, "học phí"))["done"]["found"] is True


def test_rewrite_is_skipped_for_a_self_contained_question():
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    assert not needs_rewrite("Điều kiện được công nhận tốt nghiệp đại học là gì?", history)
    assert needs_rewrite("còn vừa làm vừa học thì sao?", history)
    assert needs_rewrite("bao nhiêu?", history)
    assert not needs_rewrite("bất kỳ câu nào", [])


def test_rewrite_is_called_only_when_needed(settings, retriever):
    from tests.conftest import FakeLLM

    llm = FakeLLM(chunks=["ok"])
    history = [{"role": "user", "content": "thời gian đào tạo"},
               {"role": "assistant", "content": "12 học kỳ"}]
    service = ChatService(settings, retriever, llm)
    events(service, "Điều kiện được công nhận tốt nghiệp là gì?", history)
    assert llm.completions == []
    events(service, "còn hệ vừa làm vừa học thì sao?", history)
    assert len(llm.completions) == 1


# --- the rewrite guard ------------------------------------------------------
# Regression: asked "công thức nấu lẩu thái" after a turn about study duration, the model
# rewrote the question into the *previous* one and the system answered it, with citations.


def test_rewrite_that_keeps_nothing_of_the_question_is_rejected():
    assert not accept_rewrite(
        "công thức nấu lẩu thái",
        "Thời gian đào tạo tối đa cho sinh viên vừa làm vừa học là bao nhiêu học kỳ chính?")


def test_legitimate_rewrite_is_accepted():
    assert accept_rewrite(
        "còn vừa làm vừa học thì sao?",
        "Thời gian đào tạo tối đa cho sinh viên vừa làm vừa học là bao nhiêu học kỳ chính?")


def test_pure_deixis_has_nothing_to_preserve():
    assert accept_rewrite("vậy thì sao?", "Điều kiện buộc thôi học là gì?")


def test_empty_or_runaway_rewrite_is_rejected():
    assert not accept_rewrite("học phí", "")
    assert not accept_rewrite("học phí", "học phí " * 60)


def test_off_topic_follow_up_falls_back_to_the_raw_question(settings, retriever):
    from tests.conftest import FakeLLM

    history = [{"role": "user", "content": "thời gian đào tạo tối đa"},
               {"role": "assistant", "content": "12 học kỳ chính"}]
    hijack = "Thời gian đào tạo tối đa cho sinh viên vừa làm vừa học là bao nhiêu học kỳ chính?"
    llm = FakeLLM(reply=hijack, chunks=["ok"])
    service = ChatService(settings, retriever, llm)
    names = dict(events(service, "công thức lẩu thái", history))
    # the rewrite was attempted, rejected, and never announced to the client
    assert llm.completions
    assert "query" not in names
    # and the excerpts come from searching what the user actually asked
    assert all("lẩu" not in str(s.get("label")) for s in (names.get("sources") or []))


def test_a_five_word_question_no_longer_triggers_a_rewrite():
    """The question from the bug report: self-contained, just short."""
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    assert not needs_rewrite("công thức nấu lẩu thái", history)
