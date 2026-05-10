"""Dependency and path checks for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Checks local dependencies needed by training stages.

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import ValidationResult
from orac_wake_lab.services.training_config import (
    validate_training_config_inputs,
)


def run_dependency_checks(project: WakeWordProject) -> list[ValidationResult]:
    """Run Phase 1 dependency and path checks.

    Args:
        project (WakeWordProject): Project settings to validate.

    Returns:
        list[ValidationResult]: Structured check results.
    """
    results = [
        _check_python(),
        _check_repo_path(project.openwakeword_repo),
        _check_openwakeword_import(project.openwakeword_repo),
        _check_import("yaml", ["generate", "augment", "train"]),
        _check_import("torch", ["generate", "augment", "train"]),
        _check_import("torchaudio", ["augment", "train"]),
        _check_import("piper", ["generate", "augment", "train"]),
        _check_import("tensorflow", ["convert_tflite"]),
        _check_import("onnx", ["convert_tflite"]),
        _check_import("onnx_tf", ["convert_tflite"]),
        _check_piper_generator(project.piper_sample_generator_path),
        _check_directories(
            "Background audio directories",
            project.background_paths or [],
            ["augment", "train"],
        ),
        _check_directories(
            "RIR directories",
            project.rir_paths or [],
            ["augment", "train"],
        ),
        _check_validation_data(project.false_positive_validation_data_path),
        validate_training_config_inputs(project),
    ]
    return results


def _check_python() -> ValidationResult:
    return ValidationResult(
        name="Python executable",
        status="pass",
        message=sys.executable,
        blocks=[],
    )


def _check_repo_path(path: Path) -> ValidationResult:
    if path.exists() and (path / "openwakeword" / "train.py").exists():
        return ValidationResult(
            name="openWakeWord repository",
            status="pass",
            message=str(path),
            blocks=[],
        )
    return ValidationResult(
        name="openWakeWord repository",
        status="fail",
        message=f"Repository path is not usable: {path}",
        blocks=["generate", "augment", "train", "convert_tflite"],
    )


def _check_import(module_name: str, blocks: list[str]) -> ValidationResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return ValidationResult(
            name=f"import {module_name}",
            status="fail",
            message=str(exc),
            blocks=blocks,
        )
    return ValidationResult(
        name=f"import {module_name}",
        status="pass",
        message="Import succeeded.",
        blocks=[],
    )


def _check_openwakeword_import(repo_path: Path) -> ValidationResult:
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(repo_path))
        importlib.import_module("openwakeword")
    except Exception as exc:
        return ValidationResult(
            name="import openwakeword",
            status="fail",
            message=str(exc),
            blocks=["generate", "augment", "train"],
        )
    finally:
        sys.path = original_path
    return ValidationResult(
        name="import openwakeword",
        status="pass",
        message="Import succeeded using the configured repository path.",
        blocks=[],
    )


def _check_piper_generator(path: Path) -> ValidationResult:
    if not str(path):
        return ValidationResult(
            name="Piper sample generator",
            status="fail",
            message="No Piper sample generator path configured.",
            blocks=["generate", "augment", "train", "convert_tflite"],
        )
    
    main_py = path / "piper_sample_generator" / "__main__.py"
    augment_py = path / "piper_sample_generator" / "augment.py"
    
    if not path.exists() or not main_py.exists() or not augment_py.exists():
        return ValidationResult(
            name="Piper sample generator",
            status="fail",
            message=f"Missing piper_sample_generator module or expected files under {path}",
            blocks=["generate", "augment", "train", "convert_tflite"],
        )

    return ValidationResult(
        name="Piper sample generator",
        status="pass",
        message=str(path),
        blocks=[],
    )


def _check_directories(
    name: str,
    paths: list[Path],
    blocks: list[str],
) -> ValidationResult:
    if not paths:
        return ValidationResult(
            name=name,
            status="fail",
            message="No directories configured.",
            blocks=blocks,
        )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return ValidationResult(
            name=name,
            status="fail",
            message="Missing: " + ", ".join(missing),
            blocks=blocks,
        )
    return ValidationResult(
        name=name,
        status="pass",
        message="All configured directories exist.",
        blocks=[],
    )


def _check_validation_data(path: Path) -> ValidationResult:
    if not str(path):
        return ValidationResult(
            name="False-positive validation data",
            status="fail",
            message="No .npy validation feature file configured.",
            blocks=["train"],
        )
    if path.exists() and path.suffix == ".npy":
        return ValidationResult(
            name="False-positive validation data",
            status="pass",
            message=str(path),
            blocks=[],
        )
    return ValidationResult(
        name="False-positive validation data",
        status="fail",
        message=f"Missing .npy validation feature file: {path}",
        blocks=["train"],
    )
