# pipeline/logging_conf.py
import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

def setup_logger(log_file: Path | None = None) -> logging.Logger:
    """Configure root logger to write to stdout (and optional file)."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,   # override previous configs
    )
    return logging.getLogger(__name__)
