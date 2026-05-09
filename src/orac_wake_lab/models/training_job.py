"""Training job models for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Defines subprocess training job state.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class JobStatus(StrEnum):
    """Supported training job states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Represent a single training subprocess job.

    Args:
        name (str): Stage name.
        command (list[str]): Command and arguments.
        cwd (Path): Working directory.
        log_path (Path): Log file path.
        status (JobStatus): Current job status.
        return_code (int | None): Subprocess return code.
    """

    name: str
    command: list[str]
    cwd: Path
    log_path: Path
    status: JobStatus = JobStatus.PENDING
    return_code: int | None = None
