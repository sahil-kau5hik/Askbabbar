import json
import pytest
from pathlib import Path
from backend.ingestion.downloader import (
    get_audio_filepath,
    is_audio_already_downloaded,
    update_manifest_download_status,
)
from backend.models.schemas import VideoMetadata


def test_get_audio_filepath(tmp_path: Path):
    path = get_audio_filepath("test12345", output_dir=tmp_path)
    assert path == tmp_path / "test12345.mp3"
    assert tmp_path.exists()


def test_is_audio_already_downloaded(tmp_path: Path):
    vid = "abc_check"
    assert is_audio_already_downloaded(vid, output_dir=tmp_path) is False

    # Create dummy empty file
    file_path = tmp_path / f"{vid}.mp3"
    file_path.write_bytes(b"small")
    assert is_audio_already_downloaded(vid, output_dir=tmp_path) is False

    # Create dummy file > 10KB
    file_path.write_bytes(b"0" * 15000)
    assert is_audio_already_downloaded(vid, output_dir=tmp_path) is True


def test_update_manifest_download_status(tmp_path: Path):
    manifest_file = tmp_path / "pipeline_manifest.json"
    dummy_manifest = {
        "vid_999": {
            "video_id": "vid_999",
            "title": "Test Lecture",
            "downloaded": False,
            "audio_path": None,
            "error": None
        }
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(dummy_manifest, f)

    update_manifest_download_status(
        video_id="vid_999",
        success=True,
        audio_path=str(tmp_path / "vid_999.mp3"),
        manifest_path=manifest_file
    )

    with open(manifest_file, "r", encoding="utf-8") as f:
        updated = json.load(f)

    assert updated["vid_999"]["downloaded"] is True
    assert updated["vid_999"]["audio_path"] == str(tmp_path / "vid_999.mp3")
    assert updated["vid_999"]["error"] is None
