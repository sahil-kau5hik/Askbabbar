import logging
import sys
from typing import Optional

def setup_logger(name: str = "love_babbar_rag", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger for the pipeline."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()
