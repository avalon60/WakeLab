"""Validation result models for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Defines structured validation and dependency check results.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


ValidationStatus = Literal["pass", "warn", "fail"]
MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class ValidationResult:
    """Represent a validation or dependency check result."""

    name: str
    status: ValidationStatus
    message: str
    blocks: list[str] = field(default_factory=list)

    @property
    def is_failure(self) -> bool:
        """Return whether the validation failed."""
        return self.status == "fail"


def derive_model_name(wake_phrase: str) -> str:
    """Derive a model-safe name from a wake phrase.

    Args:
        wake_phrase (str): Wake phrase.

    Returns:
        str: Lowercase underscore-separated model name.
    """
    words = re.findall(r"[a-z0-9]+", wake_phrase.lower())
    return "_".join(words)


def validate_phrase(wake_phrase: str) -> ValidationResult:
    """Validate a wake phrase.

    Args:
        wake_phrase (str): Wake phrase to validate.

    Returns:
        ValidationResult: Pass, warning, or failure result.
    """
    cleaned = wake_phrase.strip()
    if not cleaned:
        return ValidationResult(
            name="Wake phrase",
            status="fail",
            message="Wake phrase cannot be empty.",
            blocks=["project"],
        )
    if len(cleaned.split()) < 2 or len(cleaned) < 6:
        return ValidationResult(
            name="Wake phrase",
            status="warn",
            message="Short phrases may be harder to train robustly.",
            blocks=[],
        )
    return ValidationResult(
        name="Wake phrase",
        status="pass",
        message="Wake phrase looks usable.",
        blocks=[],
    )


def validate_model_name(model_name: str) -> ValidationResult:
    """Validate a model name.

    Args:
        model_name (str): Model name to validate.

    Returns:
        ValidationResult: Validation result.
    """
    if not model_name.strip():
        return ValidationResult(
            name="Model name",
            status="fail",
            message="Model name cannot be empty.",
            blocks=["project"],
        )
    if MODEL_NAME_PATTERN.fullmatch(model_name) is None:
        return ValidationResult(
            name="Model name",
            status="fail",
            message="Use lowercase letters, numbers, and underscores only.",
            blocks=["project"],
        )
    return ValidationResult(
        name="Model name",
        status="pass",
        message="Model name is valid.",
        blocks=[],
    )
