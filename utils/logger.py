"""
utils/logger.py
---------------
One small helper, `get_logger(name)`, that every module uses to get a
consistent logger.

Why bother with logging (instead of print)?
  * You can switch the verbosity globally (LOG_LEVEL in settings).
  * Logs go to BOTH the console and a rotating file in logs/, so you have a
    persistent record of what the backend and simulator did.
  * Each line is timestamped and tagged with the module name, which makes
    debugging a running system far easier than scattered prints.

"Rotating" file handler = when the log file reaches a size cap it is renamed
(app.log.1, app.log.2 …) and a fresh file starts, so logs never grow forever.
"""

import logging
from logging.handlers import RotatingFileHandler

from core.settings import settings

# Build the log line format once: time | LEVEL | module | message
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Track which logger names we've already configured, so repeated calls to
# get_logger() don't attach duplicate handlers (which would double-print).
_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and logs/app.log."""
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    logger.setLevel(settings.LOG_LEVEL.upper())
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    # 1) Console handler — what you see in the terminal.
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 2) Rotating file handler — persistent history (max 2 MB × 3 backups).
    log_file = settings.LOG_DIR / "app.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't let messages bubble up to the root logger and print twice.
    logger.propagate = False

    _configured.add(name)
    return logger
