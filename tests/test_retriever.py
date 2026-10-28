import pytest
from backend.models.schemas import QueryRequest
from backend.retrieval.retriever import embed_query, search


def test_embed_query_dimension():
    vector = embed_query("binary search algorithm")
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


def test_retriever_search_existing_collection():
    request = QueryRequest(
        query="binary search",
        top_k=3,
        similarity_threshold=0.10,
        expand_adjacent=True,
    )
    results = search(request)
    assert len(results) > 0
    assert len(results) <= 3
    assert results[0].relevance_score >= 0.10
    assert results[0].youtube_url.startswith("https://www.youtube.com/watch?")
    assert results[0].lecture_number in [6, 12, 13, 14, 23, 33]


def test_retriever_search_high_threshold_empty():
    request = QueryRequest(
        query="binary search",
        top_k=3,
        similarity_threshold=0.99,
    )
    results = search(request)
    assert results == []
