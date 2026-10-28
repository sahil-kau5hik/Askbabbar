from backend.llm.generator import (
    _assess_confidence,
    _build_context_block,
    _build_user_prompt,
    generate_answer,
)
from backend.models.schemas import TopicSource


def _create_dummy_source(lecture_num: int, score: float, text: str) -> TopicSource:
    return TopicSource(
        lecture_number=lecture_num,
        title=f"Lecture {lecture_num}",
        video_id=f"vid_{lecture_num}",
        start_time=10.0,
        end_time=70.0,
        timestamp_range="00:10 - 01:10",
        youtube_url=f"https://youtube.com/watch?v=vid_{lecture_num}&t=10s",
        relevance_score=score,
        matched_transcript_snippet=text,
        adjacent_context="Next section covers examples.",
    )


def test_build_context_block_empty():
    block = _build_context_block([])
    assert "No relevant transcript excerpts found" in block


def test_build_context_block_with_sources():
    sources = [
        _create_dummy_source(12, 0.85, "Binary search requires sorted array."),
        _create_dummy_source(13, 0.65, "First and last position of an element in sorted array."),
    ]
    block = _build_context_block(sources)
    assert "--- Source 1: Lecture 12 | 00:10 - 01:10 ---" in block
    assert "Binary search requires sorted array." in block
    assert "[Adjacent context]: Next section covers examples." in block
    assert "--- Source 2: Lecture 13 | 00:10 - 01:10 ---" in block


def test_build_user_prompt():
    sources = [_create_dummy_source(12, 0.85, "mid = s + (e-s)/2")]
    prompt = _build_user_prompt("How to avoid integer overflow in mid?", sources)
    assert "QUESTION: How to avoid integer overflow in mid?" in prompt
    assert "mid = s + (e-s)/2" in prompt


def test_assess_confidence():
    assert _assess_confidence([], "Any answer") == "low"

    high_source = [_create_dummy_source(1, 0.78, "sample")]
    assert _assess_confidence(high_source, "answer") == "high"

    med_source = [_create_dummy_source(1, 0.55, "sample")]
    assert _assess_confidence(med_source, "answer") == "medium"

    low_source = [_create_dummy_source(1, 0.32, "sample")]
    assert _assess_confidence(low_source, "answer") == "low"


def test_generate_answer_extractive_fallback(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    sources = [_create_dummy_source(12, 0.75, "Binary Search")]
    resp = generate_answer("What is binary search?", sources)
    assert resp.query == "What is binary search?"
    assert len(resp.sources) == 1
    assert "Displaying grounded context directly from lecture transcripts" in resp.answer
    assert "Lecture 12" in resp.answer


def test_generate_answer_invalid_provider_graceful(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "LLM_API_KEY", "dummy_test_key")
    sources = [_create_dummy_source(12, 0.75, "Binary Search")]
    resp = generate_answer("What is binary search?", sources, provider="non_existent_provider")
    assert resp.query == "What is binary search?"
    assert len(resp.sources) == 1
    assert "Sorry, I encountered an error" in resp.answer
