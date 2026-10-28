#!/usr/bin/env python3
"""
Phase 3 Ingestion Script: Whisper Transcription
Transcribes extracted audio tracks into timestamped segments using faster-whisper.
Saves transcripts to data/transcripts/{video_id}.json and updates data/manifests/pipeline_manifest.json.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.ingestion.playlist import load_cached_videos
from backend.ingestion.transcriber import transcribe_batch
from backend.utils.logging import logger


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Transcribe Love Babbar DSA lecture audio with timestamps"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to transcribe"
    )
    parser.add_argument(
        "--lectures",
        type=int,
        nargs="+",
        default=None,
        help="Specific lecture numbers to transcribe (e.g. --lectures 12)"
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific video IDs to transcribe"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.WHISPER_MODEL,
        help=f"Whisper model size (default: '{settings.WHISPER_MODEL}')"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-transcription even if transcript already exists"
    )
    args = parser.parse_args()

    logger.info("=== Starting Phase 3: Whisper Transcription ===")
    all_videos = load_cached_videos()

    if not all_videos:
        logger.error("No videos found in cache! Run Phase 1 first.")
        sys.exit(1)

    target_videos = all_videos

    if args.lectures:
        target_set = set(args.lectures)
        target_videos = [v for v in all_videos if v.lecture_number in target_set]
        logger.info(f"Filtered by lecture numbers {args.lectures}: {len(target_videos)} video(s) found.")

    elif args.video_ids:
        target_set = set(args.video_ids)
        target_videos = [v for v in all_videos if v.video_id in target_set]
        logger.info(f"Filtered by video IDs {args.video_ids}: {len(target_videos)} video(s) found.")

    if args.limit and args.limit > 0:
        target_videos = target_videos[:args.limit]
        logger.info(f"Applying limit: transcribing first {args.limit} video(s).")

    # Filter to only videos whose audio file is downloaded
    available_videos = [v for v in target_videos if (settings.audio_dir / f"{v.video_id}.mp3").exists()]
    logger.info(f"Videos with downloaded audio available for transcription: {len(available_videos)}/{len(target_videos)}")
    target_videos = available_videos

    if not target_videos:
        logger.warning("No matching videos found with downloaded audio to transcribe.")
        sys.exit(0)

    results = transcribe_batch(
        videos=target_videos,
        force_retranscribe=args.force,
        model_size=args.model
    )

    logger.info("=== Phase 3 Summary ===")
    logger.info(f"Total Target Lectures: {results['total']}")
    logger.info(f"Newly Transcribed:    {results['transcribed']}")
    logger.info(f"Already Cached:       {results['skipped']}")
    logger.info(f"Failed Transcriptions: {results['failed']}")
    if results['failed_ids']:
        logger.error(f"Failed Video IDs: {results['failed_ids']}")

    logger.info("Phase 3 finished successfully!")


if __name__ == "__main__":
    main()
