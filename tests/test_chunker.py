import json
import pytest
from pathlib import Path
from backend.ingestion.chunker import (
    chunk_transcript,
    save_chunks_file,
    load_chunks_file,
    update_manifest_chunk_status,
)
from backend.models.schemas import TranscriptChunk, VideoMetadata, VideoTranscript, WhisperSegment


def test_chunk_transcript_boundary_preservation():
    video_meta = VideoMetadata(
        video_id="chunk_test_vid",
        title="Lecture 12: Binary Search",
        url="https://www.youtube.com/watch?v=chunk_test_vid",
        playlist_index=12,
        lecture_number=12
    )

    # 4 segments: 0-25s, 25-55s, 55-75s, 75-130s
    segments = [
        WhisperSegment(id=0, start=0.0, end=25.4, text="Welcome to Binary Search."),
        WhisperSegment(id=1, start=25.4, end=55.1, text="It works on sorted arrays."),
        WhisperSegment(id=2, start=55.1, end=75.8, text="Here is the middle index formula."),
        WhisperSegment(id=3, start=75.8, end=130.2, text="Time complexity is logarithmic.")
    ]

    transcript = VideoTranscript(
        video_id="chunk_test_vid",
        lecture_number=12,
        video_title="Lecture 12: Binary Search",
        total_duration=130.2,
        segments=segments
    )

    chunks = chunk_transcript(transcript, video_meta, target_duration=60.0)

    assert len(chunks) == 2

    # First chunk should cover seg 0, 1, 2 (0.0s to 75.8s >= 60s)
    c1 = chunks[0]
    assert c1.start_time == 0.0
    assert c1.end_time == 75.8
    assert c1.start_timestamp_str == "00:00"
    assert c1.end_timestamp_str == "01:15"
    assert c1.youtube_timestamp_url == "https://www.youtube.com/watch?v=chunk_test_vid&t=0s"
    assert c1.chunk_id == "chunk_test_vid_000000_000075"
    assert "Welcome to Binary Search." in c1.text
    assert "middle index formula." in c1.text

    # Second chunk should cover seg 3
    c2 = chunks[1]
    assert c2.start_time == 75.8
    assert c2.end_time == 130.2
    assert c2.start_timestamp_str == "01:15"
    assert c2.end_timestamp_str == "02:10"
    assert c2.youtube_timestamp_url == "https://www.youtube.com/watch?v=chunk_test_vid&t=75s"
    assert "logarithmic" in c2.text


def test_save_and_load_chunks(tmp_path: Path):
    chunks = [
        TranscriptChunk(
            chunk_id="v1_000000_000060",
            video_id="v1",
            video_title="Lec 1",
            lecture_number=1,
            playlist_index=1,
            video_url="https://youtube.com/watch?v=v1",
            start_time=0.0,
            end_time=60.5,
            start_timestamp_str="00:00",
            end_timestamp_str="01:00",
            youtube_timestamp_url="https://youtube.com/watch?v=v1&t=0s",
            text="Intro text"
        )
    ]

    saved_path = save_chunks_file(chunks, "v1", output_dir=tmp_path)
    assert saved_path.exists()

    loaded = load_chunks_file("v1", chunks_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].chunk_id == "v1_000000_000060"
    assert loaded[0].start_time == 0.0
    assert loaded[0].end_time == 60.5


def test_update_manifest_chunk_status(tmp_path: Path):
    manifest_file = tmp_path / "pipeline_manifest.json"
    dummy = {
        "v_chunk": {
            "video_id": "v_chunk",
            "title": "Chunk Lecture",
            "chunked": False,
            "chunks_path": None,
            "error": None
        }
    }
    manifest_file.write_text(json.dumps(dummy), encoding="utf-8")

    update_manifest_chunk_status(
        video_id="v_chunk",
        success=True,
        chunks_path=str(tmp_path / "v_chunk.json"),
        manifest_path=manifest_file
    )

    updated = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert updated["v_chunk"]["chunked"] is True
    assert updated["v_chunk"]["chunks_path"] == str(tmp_path / "v_chunk.json")
