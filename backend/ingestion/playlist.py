import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yt_dlp
from backend.config import settings
from backend.models.schemas import PipelineManifestEntry, VideoMetadata
from backend.utils.logging import logger

# Regex to detect lecture number in titles: "Lecture 12", "Lec 12", "Lec-12", "#12", etc.
LECTURE_NUM_REGEX = re.compile(
    r"(?:lecture|lec|episode|ep|#)[\s\-_:]*(\d+)",
    re.IGNORECASE
)


def extract_lecture_number(title: str, fallback_index: Optional[int] = None) -> Optional[int]:
    """
    Extracts the lecture number from the video title.
    Falls back to playlist index if no number pattern is matched.
    """
    if not title:
        return fallback_index

    match = LECTURE_NUM_REGEX.search(title)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            pass

    return fallback_index


def is_invalid_video(entry: Dict) -> bool:
    """Checks if a playlist entry is deleted, private, or missing essential IDs."""
    if not entry:
        return True
    
    title = (entry.get("title") or "").strip().lower()
    if title in ["[deleted video]", "[private video]", "deleted video", "private video"]:
        return True
    
    video_id = entry.get("id") or entry.get("video_id")
    if not video_id or video_id == "":
        return True

    return False


def fetch_playlist_metadata(
    playlist_url: Optional[str] = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    force_refresh: bool = False
) -> List[VideoMetadata]:
    """
    Fetches flat playlist metadata using yt-dlp.
    Handles network retries, duplicate video removal, and invalid/deleted videos.
    If cached metadata exists and force_refresh is False, loads from cache.
    """
    url = playlist_url or settings.PLAYLIST_URL
    cache_path = settings.videos_json_path

    if not force_refresh and cache_path.exists():
        logger.info(f"Loading cached playlist metadata from {cache_path}")
        try:
            cached_videos = load_cached_videos(cache_path)
            if cached_videos:
                logger.info(f"Successfully loaded {len(cached_videos)} videos from cache.")
                return cached_videos
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}. Re-fetching playlist...")

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
    }

    logger.info(f"Connecting to YouTube to fetch playlist metadata: {url}")
    raw_info = None

    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                raw_info = ydl.extract_info(url, download=False)
                if raw_info and "entries" in raw_info:
                    break
        except Exception as exc:
            logger.warning(
                f"[Attempt {attempt}/{max_retries}] Error fetching playlist from {url}: {exc}"
            )
            if attempt < max_retries:
                sleep_time = backoff_factor ** attempt
                logger.info(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Exhausted all {max_retries} retries for playlist {url}")
                raise

    if not raw_info or "entries" not in raw_info:
        raise ValueError(f"Could not extract entries from playlist: {url}")

    playlist_title = raw_info.get("title") or "Complete C++ Placement DSA Course"
    channel_name = raw_info.get("uploader") or raw_info.get("channel") or "CodeHelp - by Babbar"
    entries = raw_info.get("entries") or []

    logger.info(f"Playlist Title: '{playlist_title}' | Total Raw Entries: {len(entries)}")

    videos: List[VideoMetadata] = []
    seen_video_ids: Set[str] = set()

    for idx, entry in enumerate(entries, start=1):
        if is_invalid_video(entry):
            logger.warning(f"Skipping invalid/deleted/private video at index {idx}: {entry}")
            continue

        video_id = entry.get("id")
        if video_id in seen_video_ids:
            logger.warning(f"Duplicate video ID '{video_id}' detected at index {idx}. Skipping.")
            continue
        seen_video_ids.add(video_id)

        raw_title = entry.get("title", f"Lecture {idx}").strip()
        lecture_num = extract_lecture_number(raw_title, fallback_index=idx)
        duration = entry.get("duration")
        if duration is not None:
            try:
                duration = int(duration)
            except (ValueError, TypeError):
                duration = None

        video_url = entry.get("url")
        if not video_url or not video_url.startswith("http"):
            video_url = f"https://www.youtube.com/watch?v={video_id}"

        upload_date = entry.get("upload_date")
        if not upload_date and entry.get("timestamp"):
            try:
                upload_date = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc).strftime("%Y%m%d")
            except Exception:
                upload_date = None

        meta = VideoMetadata(
            video_id=video_id,
            title=raw_title,
            url=video_url,
            duration=duration,
            playlist_index=idx,
            playlist_title=playlist_title,
            channel=channel_name,
            upload_date=upload_date,
            lecture_number=lecture_num,
        )
        videos.append(meta)

    logger.info(f"Successfully processed {len(videos)} valid unique lectures from playlist.")
    return videos


def save_videos_metadata(videos: List[VideoMetadata], save_path: Optional[Path] = None) -> Path:
    """Saves validated video metadata to JSON."""
    dest = save_path or settings.videos_json_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = [v.model_dump() for v in videos]
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(videos)} video metadata records to {dest}")
    return dest


def load_cached_videos(cache_path: Optional[Path] = None) -> List[VideoMetadata]:
    """Loads video metadata from cached JSON file."""
    src = cache_path or settings.videos_json_path
    if not src.exists():
        return []

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [VideoMetadata(**item) for item in data]


def sync_pipeline_manifest(
    videos: List[VideoMetadata],
    manifest_path: Optional[Path] = None
) -> Dict[str, Dict]:
    """
    Initializes or updates the pipeline status manifest (`pipeline_manifest.json`).
    Preserves existing progress (downloaded, transcribed, chunked, uploaded) for previously seen videos.
    """
    dest = manifest_path or settings.pipeline_manifest_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing_manifest: Dict[str, Dict] = {}
    if dest.exists():
        try:
            with open(dest, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing manifest: {e}. Starting fresh.")

    now_iso = datetime.now(timezone.utc).isoformat()
    updated_manifest: Dict[str, Dict] = {}

    for video in videos:
        v_id = video.video_id
        if v_id in existing_manifest:
            entry = existing_manifest[v_id]
            # Update title, index or lecture_number if changed
            entry["title"] = video.title
            entry["playlist_index"] = video.playlist_index
            entry["lecture_number"] = video.lecture_number
            updated_manifest[v_id] = entry
        else:
            new_entry = PipelineManifestEntry(
                video_id=v_id,
                title=video.title,
                playlist_index=video.playlist_index,
                lecture_number=video.lecture_number,
                downloaded=False,
                transcribed=False,
                chunked=False,
                embedded=False,
                uploaded_to_qdrant=False,
                audio_path=str(settings.audio_dir / f"{v_id}.mp3"),
                transcript_path=str(settings.transcripts_dir / f"{v_id}.json"),
                chunks_path=str(settings.chunks_dir / f"{v_id}.json"),
                last_updated=now_iso,
                error=None,
            )
            updated_manifest[v_id] = new_entry.model_dump()

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Pipeline manifest synchronized at {dest} ({len(updated_manifest)} total entries).")
    return updated_manifest
