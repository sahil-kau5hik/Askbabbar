import os
import shutil
from pathlib import Path
from typing import Optional

from backend.utils.logging import logger

_FFMPEG_PATH: Optional[str] = None


def get_ffmpeg_binary() -> str:
    """
    Locates FFmpeg executable.
    Checks:
    1. System PATH
    2. imageio-ffmpeg bundled binary (creates ffmpeg.exe copy if named with version)
    Configures process PATH dynamically so yt-dlp and child processes locate 'ffmpeg'.
    """
    global _FFMPEG_PATH
    if _FFMPEG_PATH and Path(_FFMPEG_PATH).exists():
        return _FFMPEG_PATH

    # 1. System PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _FFMPEG_PATH = system_ffmpeg
        return _FFMPEG_PATH

    # 2. imageio-ffmpeg
    try:
        import imageio_ffmpeg
        bundled_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_exe and Path(bundled_exe).exists():
            src_path = Path(bundled_exe)
            target_ffmpeg = src_path.parent / "ffmpeg.exe"
            
            # If the binary has a version suffix (e.g. ffmpeg-win-x86_64-v7.1.exe), ensure ffmpeg.exe exists
            if not target_ffmpeg.exists():
                try:
                    shutil.copyfile(src_path, target_ffmpeg)
                    logger.info(f"Created standard ffmpeg.exe at {target_ffmpeg}")
                except Exception as copy_err:
                    logger.warning(f"Failed to copy standard ffmpeg.exe: {copy_err}")
                    target_ffmpeg = src_path

            _FFMPEG_PATH = str(target_ffmpeg)
            
            # Add directory to PATH so yt-dlp child processes locate it as 'ffmpeg'
            ffmpeg_dir = str(target_ffmpeg.parent)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"
            return _FFMPEG_PATH
    except Exception as exc:
        logger.warning(f"Failed to load imageio-ffmpeg: {exc}")

    raise RuntimeError(
        "FFmpeg binary not found. Please install FFmpeg or ensure imageio-ffmpeg is installed."
    )
