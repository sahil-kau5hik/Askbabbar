import math
from typing import Union

def format_timestamp(seconds: Union[int, float]) -> str:
    """
    Converts seconds into human-readable MM:SS or HH:MM:SS format.
    Example: 755 -> "12:35", 3665 -> "01:01:05"
    """
    if seconds is None or seconds < 0:
        return "00:00"
    
    total_seconds = int(math.floor(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_youtube_timestamp_url(video_id: str, seconds: Union[int, float]) -> str:
    """
    Builds an exact, clickable YouTube timestamp URL.
    Uses exact integer second from start time.
    Example: '70A40f_B5k4', 755.4 -> 'https://www.youtube.com/watch?v=70A40f_B5k4&t=755s'
    """
    clean_id = (video_id or "").strip()
    sec_int = max(0, int(math.floor(seconds or 0)))
    return f"https://www.youtube.com/watch?v={clean_id}&t={sec_int}s"


def parse_timestamp_to_seconds(timestamp_str: str) -> int:
    """
    Converts MM:SS or HH:MM:SS string back to total seconds.
    Example: "12:35" -> 755
    """
    parts = timestamp_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(float(timestamp_str))
