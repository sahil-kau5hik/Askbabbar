from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "embedding_model" in data
    assert "collection_info" in data
    assert data["collection_info"]["name"] == "love_babbar_dsa"


def test_collection_info_endpoint():
    response = client.get("/collection/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "love_babbar_dsa"
    assert data["points_count"] >= 1


def test_search_endpoint():
    payload = {
        "query": "binary search",
        "top_k": 2,
        "similarity_threshold": 0.10,
        "expand_adjacent": False,
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    sources = response.json()
    assert isinstance(sources, list)
    assert len(sources) <= 2
    if sources:
        first = sources[0]
        assert "youtube_url" in first
        assert "relevance_score" in first
        assert first["lecture_number"] in [6, 12, 13, 14, 23, 33]


def test_search_empty_query_422():
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 422


def test_query_empty_query_422():
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
