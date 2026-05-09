"""Log helpers for Orac Wake Lab subprocess output."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Provides simple append-only stage log writing.

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def append_log(log_path: Path, line: str) -> None:
    """Append one line to a stage log.

    Args:
        log_path (Path): Log file path.
        line (str): Line to append.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {line.rstrip()}\n")


def clear_log(log_path: Path) -> None:
    """Clear a stage log.

    Args:
        log_path (Path): Log file path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
