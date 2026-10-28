import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.models.schemas import TranscriptChunk, VideoMetadata, VideoTranscript, WhisperSegment
from backend.utils.logging import logger
from backend.utils.timestamps import build_youtube_timestamp_url, format_timestamp


def get_chunks_filepath(video_id: str, output_dir: Optional[Path] = None) -> Path:
    """Returns the deterministic path for chunked transcript JSON."""
    dest_dir = output_dir or settings.chunks_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{video_id}.json"


def is_chunk_file_already_generated(video_id: str, output_dir: Optional[Path] = None) -> bool:
    """Checks if a valid, non-empty chunks JSON exists."""
    target = get_chunks_filepath(video_id, output_dir)
    if not target.exists() or target.stat().st_size < 50:
        return False
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            return bool(isinstance(data, list) and len(data) > 0)
    except Exception:
        return False


def update_manifest_chunk_status(
    video_id: str,
    success: bool,
    chunks_path: Optional[str] = None,
    error: Optional[str] = None,
    manifest_path: Optional[Path] = None
) -> None:
    """Updates chunking status in pipeline_manifest.json."""
    m_path = manifest_path or settings.pipeline_manifest_path
    if not m_path.exists():
        return

    try:
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if video_id in manifest:
            manifest[video_id]["chunked"] = success
            if chunks_path:
                manifest[video_id]["chunks_path"] = str(chunks_path)
            manifest[video_id]["error"] = error
            manifest[video_id]["last_updated"] = datetime.now(timezone.utc).isoformat()

            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Failed to update manifest chunk status for {video_id}: {exc}")


def chunk_transcript(
    transcript: VideoTranscript,
    video_meta: VideoMetadata,
    target_duration: float = 60.0,
    min_chunk_duration: float = 15.0
) -> List[TranscriptChunk]:
    """
    Groups consecutive Whisper segments into ~60-second semantic revision windows.
    CRITICAL: Preserves Whisper's exact float timestamps without drift or approximation.
    Chunk start_time = segments[0].start
    Chunk end_time = segments[-1].end
    """
    segments = transcript.segments or []
    if not segments:
        return []

    chunks: List[TranscriptChunk] = []
    current_segments: List[WhisperSegment] = []

    for seg in segments:
        # Skip empty text
        if not seg.text or not seg.text.strip():
            continue

        current_segments.append(seg)
        current_duration = current_segments[-1].end - current_segments[0].start

        if current_duration >= target_duration:
            start_time = round(current_segments[0].start, 2)
            end_time = round(current_segments[-1].end, 2)
            merged_text = " ".join([s.text.strip() for s in current_segments]).strip()

            chunk_id = f"{video_meta.video_id}_{int(start_time):06d}_{int(end_time):06d}"
            chunks.append(
                TranscriptChunk(
                    chunk_id=chunk_id,
                    video_id=video_meta.video_id,
                    video_title=video_meta.title,
                    lecture_number=video_meta.lecture_number,
                    playlist_index=video_meta.playlist_index,
                    video_url=video_meta.url,
                    start_time=start_time,
                    end_time=end_time,
                    start_timestamp_str=format_timestamp(start_time),
                    end_timestamp_str=format_timestamp(end_time),
                    youtube_timestamp_url=build_youtube_timestamp_url(video_meta.video_id, start_time),
                    text=merged_text,
                    segment_span=[current_segments[0].id, current_segments[-1].id]
                )
            )
            current_segments = []

    # Process final leftover segments
    if current_segments:
        start_time = round(current_segments[0].start, 2)
        end_time = round(current_segments[-1].end, 2)
        duration = end_time - start_time
        merged_text = " ".join([s.text.strip() for s in current_segments]).strip()

        # If leftover is very short (< 15s) and we already have at least one chunk, merge into previous chunk
        if duration < min_chunk_duration and chunks:
            prev = chunks[-1]
            prev.end_time = end_time
            prev.end_timestamp_str = format_timestamp(end_time)
            prev.text = f"{prev.text} {merged_text}".strip()
            prev.chunk_id = f"{video_meta.video_id}_{int(prev.start_time):06d}_{int(end_time):06d}"
            if prev.segment_span:
                prev.segment_span[1] = current_segments[-1].id
        else:
            chunk_id = f"{video_meta.video_id}_{int(start_time):06d}_{int(end_time):06d}"
            chunks.append(
                TranscriptChunk(
                    chunk_id=chunk_id,
                    video_id=video_meta.video_id,
                    video_title=video_meta.title,
                    lecture_number=video_meta.lecture_number,
                    playlist_index=video_meta.playlist_index,
                    video_url=video_meta.url,
                    start_time=start_time,
                    end_time=end_time,
                    start_timestamp_str=format_timestamp(start_time),
                    end_timestamp_str=format_timestamp(end_time),
                    youtube_timestamp_url=build_youtube_timestamp_url(video_meta.video_id, start_time),
                    text=merged_text,
                    segment_span=[current_segments[0].id, current_segments[-1].id]
                )
            )

    return chunks


def save_chunks_file(chunks: List[TranscriptChunk], video_id: str, output_dir: Optional[Path] = None) -> Path:
    """Saves validated chunks list to JSON."""
    dest = get_chunks_filepath(video_id, output_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = [c.model_dump() for c in chunks]
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"[{video_id}] Saved {len(chunks)} timestamped chunks to {dest.name}")
    return dest


def load_chunks_file(video_id: str, chunks_dir: Optional[Path] = None) -> List[TranscriptChunk]:
    """Loads chunks from JSON."""
    src = get_chunks_filepath(video_id, chunks_dir)
    if not src.exists():
        return []
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TranscriptChunk(**c) for c in data]


def chunk_single_video(
    video: VideoMetadata,
    transcripts_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_rechunk: bool = False,
    target_duration: float = 60.0
) -> Tuple[bool, Optional[List[TranscriptChunk]], Optional[str]]:
    """Chunks a single video transcript into 60s windows and updates manifest."""
    v_id = video.video_id
    chunks_path = get_chunks_filepath(v_id, output_dir)

    if not force_rechunk and is_chunk_file_already_generated(v_id, output_dir):
        logger.info(f"[{v_id}] Chunks already exist in {chunks_path.name}. Skipping.")
        loaded = load_chunks_file(v_id, output_dir)
        update_manifest_chunk_status(v_id, success=True, chunks_path=str(chunks_path))
        return True, loaded, None

    t_dir = transcripts_dir or settings.transcripts_dir
    transcript_file = t_dir / f"{v_id}.json"

    if not transcript_file.exists():
        err = f"Transcript not found at {transcript_file}"
        logger.error(f"[{v_id}] {err}")
        update_manifest_chunk_status(v_id, success=False, error=err)
        return False, None, err

    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        transcript = VideoTranscript(**data)

        chunks = chunk_transcript(
            transcript=transcript,
            video_meta=video,
            target_duration=target_duration
        )

        save_chunks_file(chunks, v_id, output_dir)
        update_manifest_chunk_status(v_id, success=True, chunks_path=str(chunks_path))
        return True, chunks, None

    except Exception as exc:
        err = str(exc)
        logger.error(f"[{v_id}] Chunking failed: {err}")
        update_manifest_chunk_status(v_id, success=False, error=err)
        return False, None, err


def chunk_batch(
    videos: List[VideoMetadata],
    transcripts_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    force_rechunk: bool = False,
    target_duration: float = 60.0
) -> Dict[str, Any]:
    """Batch chunking runner across videos."""
    target_videos = videos[:limit] if limit and limit > 0 else videos
    results = {
        "total": len(target_videos),
        "chunked": 0,
        "skipped": 0,
        "failed": 0,
        "total_chunks_created": 0,
        "failed_ids": []
    }

    for idx, video in enumerate(target_videos, start=1):
        v_id = video.video_id
        if not force_rechunk and is_chunk_file_already_generated(v_id, output_dir):
            results["skipped"] += 1
            continue

        logger.info(f"Chunking ({idx}/{len(target_videos)}): [{v_id}] {video.title}")
        success, chunks, err = chunk_single_video(
            video=video,
            transcripts_dir=transcripts_dir,
            output_dir=output_dir,
            force_rechunk=force_rechunk,
            target_duration=target_duration
        )

        if success and chunks is not None:
            results["chunked"] += 1
            results["total_chunks_created"] += len(chunks)
        else:
            results["failed"] += 1
            results["failed_ids"].append({"video_id": v_id, "title": video.title, "error": err})

    return results
