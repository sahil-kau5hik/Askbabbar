import json
import pytest
from pathlib import Path
from backend.ingestion.transcriber import (
    get_transcript_filepath,
    is_transcript_already_generated,
    update_manifest_transcript_status,
)
from backend.models.schemas import VideoTranscript, WhisperSegment


def test_get_transcript_filepath(tmp_path: Path):
    path = get_transcript_filepath("vid_trans_test", output_dir=tmp_path)
    assert path == tmp_path / "vid_trans_test.json"
    assert tmp_path.exists()


def test_is_transcript_already_generated(tmp_path: Path):
    vid = "chk_trans"
    assert is_transcript_already_generated(vid, output_dir=tmp_path) is False

    target = tmp_path / f"{vid}.json"
    target.write_text("{}", encoding="utf-8")
    assert is_transcript_already_generated(vid, output_dir=tmp_path) is False

    valid_payload = {
        "video_id": vid,
        "segments": [
            {"id": 0, "start": 10.5, "end": 25.2, "text": "Testing binary search"}
        ]
    }
    target.write_text(json.dumps(valid_payload), encoding="utf-8")
    assert is_transcript_already_generated(vid, output_dir=tmp_path) is True


def test_update_manifest_transcript_status(tmp_path: Path):
    manifest_file = tmp_path / "pipeline_manifest.json"
    dummy_manifest = {
        "vid_777": {
            "video_id": "vid_777",
            "title": "Recursion Lecture",
            "transcribed": False,
            "transcript_path": None,
            "error": None
        }
    }
    manifest_file.write_text(json.dumps(dummy_manifest), encoding="utf-8")

    update_manifest_transcript_status(
        video_id="vid_777",
        success=True,
        transcript_path=str(tmp_path / "vid_777.json"),
        manifest_path=manifest_file
    )

    updated = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert updated["vid_777"]["transcribed"] is True
    assert updated["vid_777"]["transcript_path"] == str(tmp_path / "vid_777.json")
    assert updated["vid_777"]["error"] is None


def test_transcript_schema_validation():
    seg1 = WhisperSegment(id=0, start=12.4, end=25.8, text="Binary search intro")
    seg2 = WhisperSegment(id=1, start=25.8, end=40.0, text="Condition check")

    transcript = VideoTranscript(
        video_id="test_schema_vid",
        lecture_number=12,
        video_title="Binary Search",
        total_duration=120.0,
        segments=[seg1, seg2]
    )

    data = transcript.model_dump()
    assert data["video_id"] == "test_schema_vid"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["start"] == 12.4
    assert data["segments"][1]["end"] == 40.0
