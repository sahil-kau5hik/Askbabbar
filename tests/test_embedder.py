import json
from pathlib import Path
import numpy as np
import pytest
from qdrant_client import QdrantClient

from backend.embeddings.embedder import update_manifest_embed_status
from backend.embeddings.vector_store import (
    _chunk_id_to_point_id,
    _chunk_to_payload,
    ensure_collection,
    get_collection_info,
    upsert_chunks,
    delete_video_points,
    update_manifest_qdrant_status,
)
from backend.models.schemas import TranscriptChunk


@pytest.fixture
def sample_chunk() -> TranscriptChunk:
    return TranscriptChunk(
        chunk_id="vid123_000000_000060",
        video_id="vid123",
        video_title="Lecture 1: Intro to C++",
        lecture_number=1,
        playlist_index=1,
        video_url="https://youtube.com/watch?v=vid123",
        start_time=0.0,
        end_time=60.0,
        start_timestamp_str="00:00",
        end_timestamp_str="01:00",
        youtube_timestamp_url="https://youtube.com/watch?v=vid123&t=0s",
        text="Hello and welcome to DSA in C++ by Love Babbar.",
        segment_span=(0, 2),
    )


def test_chunk_id_to_point_id():
    p1 = _chunk_id_to_point_id("video_001_000010")
    p2 = _chunk_id_to_point_id("video_001_000010")
    p3 = _chunk_id_to_point_id("video_001_000020")
    assert isinstance(p1, int)
    assert p1 > 0
    assert p1 == p2
    assert p1 != p3


def test_chunk_to_payload(sample_chunk: TranscriptChunk):
    payload = _chunk_to_payload(sample_chunk)
    assert payload["chunk_id"] == "vid123_000000_000060"
    assert payload["video_id"] == "vid123"
    assert payload["lecture_number"] == 1
    assert payload["start_time"] == 0.0
    assert payload["end_time"] == 60.0
    assert payload["text"] == "Hello and welcome to DSA in C++ by Love Babbar."
    assert payload["segment_span"] == [0, 2]


def test_update_manifest_embed_status(tmp_path: Path):
    manifest_file = tmp_path / "pipeline_manifest.json"
    dummy = {
        "v_embed": {
            "video_id": "v_embed",
            "title": "Embed Lecture",
            "embedded": False,
            "error": None,
        }
    }
    manifest_file.write_text(json.dumps(dummy), encoding="utf-8")

    update_manifest_embed_status(
        video_id="v_embed",
        success=True,
        manifest_path=manifest_file,
    )

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["v_embed"]["embedded"] is True
    assert data["v_embed"]["error"] is None


def test_update_manifest_qdrant_status(tmp_path: Path):
    manifest_file = tmp_path / "pipeline_manifest.json"
    dummy = {
        "v_qdrant": {
            "video_id": "v_qdrant",
            "title": "Qdrant Lecture",
            "uploaded_to_qdrant": False,
            "error": None,
        }
    }
    manifest_file.write_text(json.dumps(dummy), encoding="utf-8")

    update_manifest_qdrant_status(
        video_id="v_qdrant",
        success=True,
        manifest_path=manifest_file,
    )

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["v_qdrant"]["uploaded_to_qdrant"] is True


def test_qdrant_upsert_and_collection_lifecycle(sample_chunk: TranscriptChunk):
    client = QdrantClient(":memory:")
    collection_name = "test_collection"
    vector_dim = 4

    ensure_collection(collection_name, vector_dim=vector_dim, client=client)

    info = get_collection_info(collection_name=collection_name, client=client)
    assert info is not None
    assert info["name"] == collection_name
    assert info["points_count"] == 0

    vectors = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    success, err = upsert_chunks(
        chunks=[sample_chunk],
        vectors=vectors,
        collection_name=collection_name,
        client=client,
    )
    assert success is True
    assert err is None

    info_after = get_collection_info(collection_name=collection_name, client=client)
    assert info_after["points_count"] == 1

    delete_video_points(
        video_id=sample_chunk.video_id,
        collection_name=collection_name,
        client=client,
    )

    info_deleted = get_collection_info(collection_name=collection_name, client=client)
    assert info_deleted["points_count"] == 0
