import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yt_dlp
from backend.config import settings
from backend.models.schemas import VideoMetadata
from backend.utils.ffmpeg import get_ffmpeg_binary
from backend.utils.logging import logger


def get_audio_filepath(video_id: str, output_dir: Optional[Path] = None, ext: str = "mp3") -> Path:
    """Returns the deterministic path for an audio file keyed strictly by video_id."""
    dest_dir = output_dir or settings.audio_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{video_id}.{ext}"


def is_audio_already_downloaded(video_id: str, output_dir: Optional[Path] = None) -> bool:
    """Checks if audio file exists on disk and is non-empty (>10KB)."""
    target = get_audio_filepath(video_id, output_dir)
    return target.exists() and target.stat().st_size > 10240


def update_manifest_download_status(
    video_id: str,
    success: bool,
    audio_path: Optional[str] = None,
    error: Optional[str] = None,
    manifest_path: Optional[Path] = None
) -> None:
    """Updates download status in pipeline_manifest.json."""
    m_path = manifest_path or settings.pipeline_manifest_path
    if not m_path.exists():
        return

    try:
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if video_id in manifest:
            manifest[video_id]["downloaded"] = success
            if audio_path:
                manifest[video_id]["audio_path"] = str(audio_path)
            manifest[video_id]["error"] = error
            manifest[video_id]["last_updated"] = datetime.now(timezone.utc).isoformat()

            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Failed to update manifest for {video_id}: {exc}")


def download_audio_for_video(
    video: VideoMetadata,
    output_dir: Optional[Path] = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    force_redownload: bool = False
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Downloads and extracts 16kHz mono MP3 audio for a single video using yt-dlp + FFmpeg.
    Returns: (success: bool, audio_path: Optional[Path], error_message: Optional[str])
    """
    video_id = video.video_id
    target_path = get_audio_filepath(video_id, output_dir)

    if not force_redownload and is_audio_already_downloaded(video_id, output_dir):
        logger.info(f"[{video_id}] Audio already exists ({target_path.stat().st_size / 1024 / 1024:.2f} MB). Skipping download.")
        update_manifest_download_status(video_id, success=True, audio_path=str(target_path))
        return True, target_path, None

    ffmpeg_bin = get_ffmpeg_binary()
    ffmpeg_dir = str(Path(ffmpeg_bin).parent)

    # yt-dlp options optimized for Whisper speech-to-text:
    # 16kHz, mono, 64kbps MP3 provides optimal Whisper quality at minimal disk footprint
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(target_path.with_suffix("")),  # yt-dlp appends .mp3 after postprocessing
        "ffmpeg_location": ffmpeg_dir,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
        "postprocessor_args": [
            "-ar", "16000",   # 16kHz sample rate standard for Whisper
            "-ac", "1",       # mono channel
        ],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "retries": 3,
        "socket_timeout": 30,
    }

    url = video.url or f"https://www.youtube.com/watch?v={video_id}"
    logger.info(f"Downloading audio for Lecture {video.lecture_number or '?'}: '{video.title}' [{video_id}]...")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if target_path.exists() and target_path.stat().st_size > 1024:
                file_mb = target_path.stat().st_size / (1024 * 1024)
                logger.info(f"[{video_id}] Download & conversion successful: {target_path.name} ({file_mb:.2f} MB)")
                update_manifest_download_status(video_id, success=True, audio_path=str(target_path))
                return True, target_path, None
            else:
                raise FileNotFoundError(f"Output audio file {target_path} was not created or is empty.")

        except Exception as exc:
            last_error = str(exc)
            logger.warning(f"[{video_id}] [Attempt {attempt}/{max_retries}] Download failed: {exc}")
            if attempt < max_retries:
                sleep_s = backoff_factor ** attempt
                time.sleep(sleep_s)

    logger.error(f"[{video_id}] Failed to download after {max_retries} attempts. Error: {last_error}")
    update_manifest_download_status(video_id, success=False, error=last_error)
    return False, None, last_error


def download_audio_batch(
    videos: List[VideoMetadata],
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    force_redownload: bool = False,
    polite_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Downloads audio for a collection of videos with progress logging and polite request intervals.
    """
    target_videos = videos[:limit] if limit and limit > 0 else videos
    total = len(target_videos)
    logger.info(f"Starting batch audio download for {total} lectures...")

    results = {
        "total": total,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "failed_ids": []
    }

    for idx, video in enumerate(target_videos, start=1):
        v_id = video.video_id
        if not force_redownload and is_audio_already_downloaded(v_id, output_dir):
            results["skipped"] += 1
            continue

        logger.info(f"Processing ({idx}/{total}): [{v_id}] {video.title}")
        success, _, err = download_audio_for_video(
            video=video,
            output_dir=output_dir,
            force_redownload=force_redownload
        )

        if success:
            results["downloaded"] += 1
        else:
            results["failed"] += 1
            results["failed_ids"].append({"video_id": v_id, "title": video.title, "error": err})

        if polite_delay > 0 and idx < total:
            time.sleep(polite_delay)

    logger.info(
        f"Batch download complete: {results['downloaded']} downloaded, "
        f"{results['skipped']} already cached, {results['failed']} failed."
    )
    return results
