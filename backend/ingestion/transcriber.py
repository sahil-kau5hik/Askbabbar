import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.models.schemas import VideoMetadata, VideoTranscript, WhisperSegment
from backend.utils.logging import logger

# PyAV v19+ compatibility patch: PyAV 19 removed the 'metadata_errors' parameter
# which faster-whisper 1.2 still passes to av.open().
try:
    import av
    _orig_av_open = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _patched_av_open
except Exception as patch_err:
    pass

# Hinglish & DSA technical vocabulary prompt biasing for Whisper
DSA_INITIAL_PROMPT = (
    "Love Babbar C++ Placement DSA Course. Topics: Binary Search, Linear Search, "
    "Arrays, Recursion, Time Complexity O(N) O(log N), Space Complexity, "
    "Linked List, Stack, Queue, Binary Tree, BST, Heap, Trie, Graph, DP, "
    "start, end, mid = start + (end - start)/2, sorted array, key, element."
)

_WHISPER_MODEL_INSTANCE = None


def get_transcript_filepath(video_id: str, output_dir: Optional[Path] = None) -> Path:
    """Returns the deterministic path for a transcript JSON file."""
    dest_dir = output_dir or settings.transcripts_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{video_id}.json"


def is_transcript_already_generated(video_id: str, output_dir: Optional[Path] = None) -> bool:
    """Checks if a valid, non-empty transcript JSON exists."""
    target = get_transcript_filepath(video_id, output_dir)
    if not target.exists() or target.stat().st_size < 50:
        return False
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            return bool(data.get("segments") and len(data["segments"]) > 0)
    except Exception:
        return False


def update_manifest_transcript_status(
    video_id: str,
    success: bool,
    transcript_path: Optional[str] = None,
    error: Optional[str] = None,
    manifest_path: Optional[Path] = None
) -> None:
    """Updates transcript generation status in pipeline_manifest.json."""
    m_path = manifest_path or settings.pipeline_manifest_path
    if not m_path.exists():
        return

    try:
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if video_id in manifest:
            manifest[video_id]["transcribed"] = success
            if transcript_path:
                manifest[video_id]["transcript_path"] = str(transcript_path)
            manifest[video_id]["error"] = error
            manifest[video_id]["last_updated"] = datetime.now(timezone.utc).isoformat()

            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Failed to update manifest transcript status for {video_id}: {exc}")


def load_whisper_model(
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None
):
    """
    Loads or returns cached faster-whisper WhisperModel instance.
    Supports GPU with automatic CPU fallback.
    """
    global _WHISPER_MODEL_INSTANCE
    if _WHISPER_MODEL_INSTANCE is not None:
        return _WHISPER_MODEL_INSTANCE

    from faster_whisper import WhisperModel

    m_size = model_size or settings.WHISPER_MODEL
    dev = device or settings.WHISPER_DEVICE
    comp = compute_type or settings.WHISPER_COMPUTE_TYPE

    threads = getattr(settings, "WHISPER_CPU_THREADS", 4)
    logger.info(f"Loading Whisper model: '{m_size}' (device: {dev}, compute_type: {comp}, cpu_threads: {threads})...")
    try:
        _WHISPER_MODEL_INSTANCE = WhisperModel(m_size, device=dev, compute_type=comp, cpu_threads=threads)
        logger.info("Whisper model loaded successfully.")
    except Exception as exc:
        if dev == "cuda":
            logger.warning(f"Failed to load Whisper on CUDA: {exc}. Falling back to CPU int8...")
            _WHISPER_MODEL_INSTANCE = WhisperModel(m_size, device="cpu", compute_type="int8", cpu_threads=threads)
        else:
            raise exc

    return _WHISPER_MODEL_INSTANCE


def transcribe_audio_file(
    audio_path: Path,
    video_id: str,
    lecture_number: Optional[int] = None,
    video_title: Optional[str] = None,
    output_dir: Optional[Path] = None,
    force_retranscribe: bool = False,
    model_size: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[bool, Optional[VideoTranscript], Optional[str]]:
    """
    Transcribes an audio file into timestamp-preserving segments using faster-whisper.
    Returns: (success: bool, transcript: Optional[VideoTranscript], error: Optional[str])
    """
    dest_path = get_transcript_filepath(video_id, output_dir)

    if not force_retranscribe and is_transcript_already_generated(video_id, output_dir):
        logger.info(f"[{video_id}] Transcript already exists. Loading from {dest_path.name}...")
        try:
            with open(dest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            transcript = VideoTranscript(**data)
            update_manifest_transcript_status(video_id, success=True, transcript_path=str(dest_path))
            return True, transcript, None
        except Exception as exc:
            logger.warning(f"Corrupt cached transcript for {video_id}: {exc}. Retranscribing...")

    if not audio_path.exists():
        err_msg = f"Audio file not found: {audio_path}"
        logger.error(f"[{video_id}] {err_msg}")
        update_manifest_transcript_status(video_id, success=False, error=err_msg)
        return False, None, err_msg

    try:
        model = load_whisper_model(model_size=model_size, device=device)
        logger.info(f"[{video_id}] Starting Whisper transcription for {audio_path.name}...")
        start_clock = time.time()

        beam = getattr(settings, "WHISPER_BEAM_SIZE", 1)
        # Run transcription with timestamp preservation and vocabulary priming
        segments_generator, info = model.transcribe(
            str(audio_path),
            beam_size=beam,
            initial_prompt=DSA_INITIAL_PROMPT,
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            word_timestamps=False, # Segment-level timestamps are robust and accurate
            vad_filter=True,       # Voice Activity Detection removes silent intervals
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments_list: List[WhisperSegment] = []
        seg_counter = 0
        for seg in segments_generator:
            clean_text = seg.text.strip()
            # Skip empty segments or segments that only consist of punctuation/symbols
            import re
            if not clean_text or not re.sub(r"[\s,.\-_!?:;]+", "", clean_text):
                continue

            segments_list.append(
                WhisperSegment(
                    id=seg_counter,
                    start=round(float(seg.start), 2),
                    end=round(float(seg.end), 2),
                    text=clean_text
                )
            )
            seg_counter += 1

        duration_sec = time.time() - start_clock
        total_audio_duration = round(float(info.duration), 2) if info and hasattr(info, "duration") else None

        logger.info(
            f"[{video_id}] Transcription complete in {duration_sec:.1f}s. "
            f"Generated {len(segments_list)} timestamped segments (Audio duration: {total_audio_duration}s)."
        )

        transcript = VideoTranscript(
            video_id=video_id,
            lecture_number=lecture_number,
            video_title=video_title,
            total_duration=total_audio_duration,
            segments=segments_list
        )

        # Save persistent JSON
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(transcript.model_dump(), f, indent=2, ensure_ascii=False)

        update_manifest_transcript_status(video_id, success=True, transcript_path=str(dest_path))
        return True, transcript, None

    except Exception as exc:
        err = str(exc)
        logger.error(f"[{video_id}] Transcription failed: {err}")
        update_manifest_transcript_status(video_id, success=False, error=err)
        return False, None, err


def transcribe_batch(
    videos: List[VideoMetadata],
    audio_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    force_retranscribe: bool = False,
    model_size: Optional[str] = None
) -> Dict[str, Any]:
    """Batch transcription runner with manifest progress tracking."""
    target_videos = videos[:limit] if limit and limit > 0 else videos
    a_dir = audio_dir or settings.audio_dir
    results = {
        "total": len(target_videos),
        "transcribed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_ids": []
    }

    for idx, video in enumerate(target_videos, start=1):
        v_id = video.video_id
        audio_file = a_dir / f"{v_id}.mp3"

        if not force_retranscribe and is_transcript_already_generated(v_id, output_dir):
            results["skipped"] += 1
            continue

        logger.info(f"Transcribing ({idx}/{len(target_videos)}): [{v_id}] {video.title}")
        success, _, err = transcribe_audio_file(
            audio_path=audio_file,
            video_id=v_id,
            lecture_number=video.lecture_number,
            video_title=video.title,
            output_dir=output_dir,
            force_retranscribe=force_retranscribe,
            model_size=model_size
        )

        if success:
            results["transcribed"] += 1
        else:
            results["failed"] += 1
            results["failed_ids"].append({"video_id": v_id, "title": video.title, "error": err})

    return results
