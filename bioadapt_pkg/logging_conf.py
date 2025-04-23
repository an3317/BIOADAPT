# pipeline/logging_conf.py
import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

def setup_logger(log_file: Path | None = None) -> logging.Logger:
    """
    Configure the root logger.

    If *log_file* is provided, the parent directory is created automatically.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        # Ensure “…/results/” (or any parent path) exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,          # override any previous config
    )
    return logging.getLogger(__name__)

