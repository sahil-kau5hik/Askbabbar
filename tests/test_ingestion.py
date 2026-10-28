import json
import pytest
from pathlib import Path
from backend.ingestion.playlist import (
    extract_lecture_number,
    is_invalid_video,
    save_videos_metadata,
    load_cached_videos,
    sync_pipeline_manifest,
)
from backend.models.schemas import VideoMetadata
from backend.utils.timestamps import format_timestamp, build_youtube_timestamp_url, parse_timestamp_to_seconds


def test_extract_lecture_number():
    assert extract_lecture_number("Lecture 1: Intro to Programming") == 1
    assert extract_lecture_number("Lecture 12: Binary Search Explained in 1 Shot") == 12
    assert extract_lecture_number("Lec 45: Reverse a Linked List") == 45
    assert extract_lecture_number("Lec-78: Hashmaps in C++") == 78
    assert extract_lecture_number("#101: Graph Algorithms") == 101
    # Fallback to playlist index
    assert extract_lecture_number("Special Doubt Session on Recursion", fallback_index=35) == 35


def test_is_invalid_video():
    assert is_invalid_video({"title": "[Deleted video]", "id": "abc"}) is True
    assert is_invalid_video({"title": "[Private video]", "id": "abc"}) is True
    assert is_invalid_video({"title": "Valid Video", "id": ""}) is True
    assert is_invalid_video({}) is True
    assert is_invalid_video({"title": "Lecture 1: Intro", "id": "WQoB2z67hvY"}) is False


def test_timestamps():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(755.4) == "12:35"
    assert format_timestamp(3665) == "01:01:05"

    assert parse_timestamp_to_seconds("12:35") == 755
    assert parse_timestamp_to_seconds("01:01:05") == 3665

    url = build_youtube_timestamp_url("WQoB2z67hvY", 755.8)
    assert url == "https://www.youtube.com/watch?v=WQoB2z67hvY&t=755s"


def test_save_and_load_metadata(tmp_path: Path):
    test_videos = [
        VideoMetadata(
            video_id="vid1",
            title="Lecture 1: Intro",
            url="https://www.youtube.com/watch?v=vid1",
            duration=3400,
            playlist_index=1,
            playlist_title="DSA Course",
            channel="CodeHelp",
            lecture_number=1,
        ),
        VideoMetadata(
            video_id="vid2",
            title="Lecture 2: Basics",
            url="https://www.youtube.com/watch?v=vid2",
            duration=2800,
            playlist_index=2,
            playlist_title="DSA Course",
            channel="CodeHelp",
            lecture_number=2,
        ),
    ]

    dest = tmp_path / "videos.json"
    save_videos_metadata(test_videos, save_path=dest)
    assert dest.exists()

    loaded = load_cached_videos(cache_path=dest)
    assert len(loaded) == 2
    assert loaded[0].video_id == "vid1"
    assert loaded[0].lecture_number == 1
    assert loaded[1].video_id == "vid2"


def test_sync_pipeline_manifest(tmp_path: Path):
    manifest_dest = tmp_path / "pipeline_manifest.json"
    test_videos = [
        VideoMetadata(
            video_id="abc123",
            title="Lecture 12: Binary Search",
            url="https://www.youtube.com/watch?v=abc123",
            duration=4200,
            playlist_index=12,
            lecture_number=12,
        )
    ]

    manifest = sync_pipeline_manifest(test_videos, manifest_path=manifest_dest)
    assert "abc123" in manifest
    assert manifest["abc123"]["downloaded"] is False
    assert manifest["abc123"]["lecture_number"] == 12

    # Verify updating existing doesn't wipe progress if already marked downloaded
    manifest["abc123"]["downloaded"] = True
    with open(manifest_dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    updated_manifest = sync_pipeline_manifest(test_videos, manifest_path=manifest_dest)
    assert updated_manifest["abc123"]["downloaded"] is True
