"""Managed standard openWakeWord feature bundle support."""
# Author: Clive Bostock
# Date: 2026-05-10
# Description: Detects, validates, registers, and downloads feature bundle files.

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import platform
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import ValidationResult
from orac_wake_lab.services import wake_lab_home


STANDARD_NEGATIVE_FEATURE_FILENAME = (
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
)
STANDARD_VALIDATION_FEATURE_FILENAME = "validation_set_features.npy"
STANDARD_FEATURE_BUNDLE_URL = (
    "https://huggingface.co/datasets/davidscripka/"
    "openwakeword_features/resolve/main"
)
STANDARD_FEATURE_BUNDLE_SIZE_NOTE = "about 17 GB"
STANDARD_FEATURE_BUNDLE_MESSAGE = (
    "WakeLab needs the standard openWakeWord feature bundle before "
    "training. These files teach the model what should not trigger the "
    "wake word. Use Download Standard Feature Bundle, Register Existing "
    "Feature Bundle, or configure paths manually in Advanced mode."
)


@dataclass(frozen=True)
class FeatureBundlePaths:
    """Represent the managed standard feature bundle paths."""

    negative: Path
    validation: Path


@dataclass(frozen=True)
class FeatureBundleCheck:
    """Represent the validation state of the managed feature bundle."""

    negative: ValidationResult
    validation: ValidationResult
    paths: FeatureBundlePaths

    @property
    def is_ready(self) -> bool:
        """Return whether both feature files are valid."""
        return not self.negative.is_failure and not self.validation.is_failure

    @property
    def status_message(self) -> str:
        """Return a plain-English bundle status message."""
        if self.is_ready:
            return (
                "Standard openWakeWord feature bundle is ready for training."
            )
        details = [
            result.message
            for result in (self.negative, self.validation)
            if result.is_failure and result.message
        ]
        if details:
            return f"{STANDARD_FEATURE_BUNDLE_MESSAGE} {' '.join(details)}"
        return STANDARD_FEATURE_BUNDLE_MESSAGE


def standard_feature_bundle_paths() -> FeatureBundlePaths:
    """Return the managed standard feature bundle file paths."""
    return FeatureBundlePaths(
        negative=(
            wake_lab_home.get_negative_features_dir()
            / STANDARD_NEGATIVE_FEATURE_FILENAME
        ),
        validation=(
            wake_lab_home.get_false_positive_validation_dir()
            / STANDARD_VALIDATION_FEATURE_FILENAME
        ),
    )


def detect_standard_feature_bundle() -> FeatureBundleCheck:
    """Validate the standard managed feature bundle in Wake Lab home."""
    paths = standard_feature_bundle_paths()
    return FeatureBundleCheck(
        negative=validate_feature_file(
            paths.negative,
            "Negative/background feature file",
        ),
        validation=validate_feature_file(
            paths.validation,
            "False-positive validation feature file",
        ),
        paths=paths,
    )


def validate_feature_file(path: Path, label: str) -> ValidationResult:
    """Validate a preprocessed feature file in plain English.

    Args:
        path (Path): Path to validate.
        label (str): Human-readable file label.

    Returns:
        ValidationResult: Structured validation result.
    """
    raw = str(path).strip()
    if raw in {"", "."}:
        return ValidationResult(
            name=label,
            status="fail",
            message=f"WakeLab does not yet have a path for the {label.lower()}.",
            blocks=["train"],
        )
    cleaned = path.expanduser()
    if cleaned.suffix != ".npy":
        return ValidationResult(
            name=label,
            status="fail",
            message=(
                f"The {label.lower()} must be a .npy file, not "
                f"{cleaned.suffix or 'an empty path'}."
            ),
            blocks=["train"],
        )
    if not cleaned.exists():
        return ValidationResult(
            name=label,
            status="fail",
            message=f"WakeLab cannot find the {label.lower()} at {cleaned}.",
            blocks=["train"],
        )
    try:
        np.load(cleaned, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return ValidationResult(
            name=label,
            status="fail",
            message=(
                f"WakeLab can see the {label.lower()}, but NumPy cannot "
                f"read it as feature data: {exc}"
            ),
            blocks=["train"],
        )
    return ValidationResult(
        name=label,
        status="pass",
        message=f"WakeLab can read the {label.lower()}.",
        blocks=[],
    )


def register_existing_feature_bundle(
    negative_source: Path,
    validation_source: Path,
) -> FeatureBundleCheck:
    """Copy an existing feature bundle into the managed Wake Lab home.

    Args:
        negative_source (Path): Existing negative feature file.
        validation_source (Path): Existing validation feature file.

    Returns:
        FeatureBundleCheck: Validation result for the managed bundle.

    Raises:
        FileNotFoundError: If either source file is missing.
        ValueError: If either source file is not a readable feature file.
    """
    negative_source = negative_source.expanduser()
    validation_source = validation_source.expanduser()
    validate_negative = validate_feature_file(
        negative_source,
        "Negative/background feature file",
    )
    validate_validation = validate_feature_file(
        validation_source,
        "False-positive validation feature file",
    )
    for result in (validate_negative, validate_validation):
        if result.is_failure:
            raise ValueError(result.message)

    paths = standard_feature_bundle_paths()
    paths.negative.parent.mkdir(parents=True, exist_ok=True)
    paths.validation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(negative_source, paths.negative)
    shutil.copy2(validation_source, paths.validation)
    return detect_standard_feature_bundle()


from typing import Callable, Optional

def download_standard_feature_bundle(
    *,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> FeatureBundleCheck:
    """Download the managed feature bundle from Hugging Face.

    Args:
        overwrite (bool): Whether to replace valid local files.
        progress_callback (Optional[Callable[[str, int, int], None]]):
            Callback receiving (filename, bytes_downloaded, total_bytes).

    Returns:
        FeatureBundleCheck: Validation result for the managed bundle.
    """
    paths = standard_feature_bundle_paths()
    paths.negative.parent.mkdir(parents=True, exist_ok=True)
    paths.validation.parent.mkdir(parents=True, exist_ok=True)

    _download_feature_file(
        STANDARD_NEGATIVE_FEATURE_FILENAME,
        paths.negative,
        overwrite=overwrite,
        progress_callback=progress_callback,
    )
    _download_feature_file(
        STANDARD_VALIDATION_FEATURE_FILENAME,
        paths.validation,
        overwrite=overwrite,
        progress_callback=progress_callback,
    )
    return detect_standard_feature_bundle()


def open_feature_bundle_folder() -> None:
    """Open the managed feature bundle folder in the file browser."""
    folder = wake_lab_home.get_features_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(folder)  # type: ignore[attr-defined]
        return
    if platform.system() == "Darwin":
        subprocess.Popen(["open", str(folder)])
        return
    subprocess.Popen(["xdg-open", str(folder)])


def apply_standard_feature_bundle_to_project(
    project: WakeWordProject,
    *,
    force: bool = False,
) -> bool:
    """Populate project feature paths from the managed bundle.

    Args:
        project (WakeWordProject): Project to update.
        force (bool): Whether to replace existing project paths.

    Returns:
        bool: Whether the project was updated.
    """
    bundle = detect_standard_feature_bundle()
    if not bundle.is_ready:
        return False
    if not force and (
        project.negative_feature_data_files
        or str(project.false_positive_validation_data_path).strip()
    ):
        return False

    project.negative_feature_data_files = {
        "negative_features": bundle.paths.negative
    }
    project.false_positive_validation_data_path = bundle.paths.validation
    return True


def _download_feature_file(
    filename: str,
    destination: Path,
    *,
    overwrite: bool,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    url = f"{STANDARD_FEATURE_BUNDLE_URL}/{filename}?download=1"
    if destination.exists() and not overwrite:
        validation = validate_feature_file(
            destination,
            "Feature bundle file",
        )
        if not validation.is_failure:
            return

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(destination.parent),
        suffix=".part",
    ) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.info().get("Content-Length", 0))
                bytes_downloaded = 0
                block_size = 1024 * 1024
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    temp_file.write(buffer)
                    bytes_downloaded += len(buffer)
                    if progress_callback:
                        progress_callback(
                            filename, bytes_downloaded, total_size
                        )
            temp_file.flush()
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    temp_path.replace(destination)
