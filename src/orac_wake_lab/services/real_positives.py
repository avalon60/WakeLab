"""Real positive wake-word clip management."""
# Author: Clive Bostock
# Date: 2026-05-15
# Description: Imports, validates, and stages user wake-word recordings.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from orac_wake_lab.models.project import WakeWordProject


TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.25
MAX_DURATION_SECONDS = 3.0
CLIPPING_RATIO_LIMIT = 0.01
SILENCE_TRIM_THRESHOLD = 400
SILENCE_KEEP_SECONDS = 0.15


@dataclass(frozen=True)
class RealPositiveValidation:
    """Validation result for a real positive WAV clip."""

    path: Path
    ok: bool
    messages: list[str]
    duration_seconds: float = 0.0
    sample_rate: int = 0


def import_real_positive_wavs(
    project: WakeWordProject,
    sources: list[Path],
) -> list[Path]:
    """Import WAV files into the project-managed real positives directory.

    Args:
        project: Wake-word project that owns the managed clip directory.
        sources: Source WAV paths selected by the user.

    Returns:
        list[Path]: Imported normalized clip paths.
    """
    destination_dir = _project_real_positive_dir(project)
    destination_dir.mkdir(parents=True, exist_ok=True)
    imported: list[Path] = []
    for source in sources:
        validation = validate_real_positive_clip(source)
        if not validation.ok:
            raise ValueError(
                f"Invalid real positive clip {source}: "
                + "; ".join(validation.messages)
            )
        destination = _next_destination_path(destination_dir, source.stem)
        audio = _read_as_mono_int16(source)
        normalized = _normalize_wake_phrase_audio(audio)
        wavfile.write(destination, TARGET_SAMPLE_RATE, normalized)
        imported.append(destination)
    return imported


def validate_real_positive_clip(path: Path) -> RealPositiveValidation:
    """Validate one candidate real positive wake-word WAV file.

    Args:
        path: WAV file to validate.

    Returns:
        RealPositiveValidation: Validation details.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return RealPositiveValidation(path, False, ["file does not exist"])
    if path.suffix.lower() != ".wav":
        errors.append("extension is not .wav")

    try:
        sample_rate, data = wavfile.read(path)
    except Exception as exc:
        return RealPositiveValidation(
            path,
            False,
            errors + [f"audio is not readable: {exc}"],
        )

    if data.size == 0:
        errors.append("audio is empty")
        return RealPositiveValidation(path, False, errors, 0.0, sample_rate)

    if data.ndim > 2:
        errors.append("audio has more than two dimensions")
    if data.ndim == 2 and data.shape[1] < 1:
        errors.append("audio has no readable channels")

    duration = len(data) / float(sample_rate) if sample_rate > 0 else 0.0
    if sample_rate <= 0:
        errors.append("sample rate is invalid")
    if duration < MIN_DURATION_SECONDS:
        errors.append("duration is too short for a wake phrase")

    mono = _to_mono_float(data)
    if np.max(np.abs(mono)) == 0:
        errors.append("audio is silent")
    if duration > MAX_DURATION_SECONDS and sample_rate > 0:
        active_duration = _active_duration_seconds(mono, sample_rate)
        if active_duration > MAX_DURATION_SECONDS:
            errors.append("duration is longer than expected for a wake phrase")
        else:
            warnings.append("excessive edge silence will be trimmed")
    if _is_clipped(data):
        errors.append("audio appears clipped")

    return RealPositiveValidation(
        path=path,
        ok=not errors,
        messages=(errors + warnings) or ["ok"],
        duration_seconds=duration,
        sample_rate=sample_rate,
    )


def validate_real_positive_directory(
    project: WakeWordProject,
) -> list[RealPositiveValidation]:
    """Validate every WAV-like file in the project's real positives folder."""
    directory = _project_real_positive_dir(project)
    if not directory.exists():
        return []
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    return [validate_real_positive_clip(path) for path in paths]


def stage_real_positives_for_training(project: WakeWordProject) -> int:
    """Normalize managed real positives into openWakeWord training input.

    Args:
        project: Project settings.

    Returns:
        int: Number of real positive training copies staged.
    """
    if not project.enable_real_positives:
        return 0
    validations = validate_real_positive_directory(project)
    failures = [
        validation for validation in validations if not validation.ok
    ]
    if failures:
        first = failures[0]
        raise ValueError(
            f"Invalid real positive clip {first.path}: "
            + "; ".join(first.messages)
        )
    if not validations:
        return 0

    positive_train_dir = (
        project.openwakeword_output_dir / project.model_name / "positive_train"
    )
    positive_train_dir.mkdir(parents=True, exist_ok=True)

    for existing in positive_train_dir.glob("real_positive_*.wav"):
        existing.unlink()

    target_percent = project.real_positive_target_percent
    synthetic_paths = [
        path
        for path in positive_train_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
        and not path.name.startswith("real_positive_")
    ]
    synthetic_count = len(synthetic_paths)
    if target_percent >= 100:
        for synthetic_path in synthetic_paths:
            synthetic_path.unlink()
    target_count = _target_real_positive_count(
        synthetic_count,
        len(validations),
        target_percent,
    )

    staged_count = 0
    for index, validation in enumerate(validations):
        audio = _read_as_mono_int16(validation.path)
        normalized = _normalize_wake_phrase_audio(audio)
        copies_for_clip = target_count // len(validations)
        if index < target_count % len(validations):
            copies_for_clip += 1
        for repeat_index in range(copies_for_clip):
            destination = positive_train_dir / (
                f"real_positive_{index:04d}_{repeat_index:02d}.wav"
            )
            wavfile.write(destination, TARGET_SAMPLE_RATE, normalized)
            staged_count += 1
    return staged_count


def real_positive_summary(project: WakeWordProject) -> str:
    """Return a short validation summary for project real positives."""
    validations = validate_real_positive_directory(project)
    if not validations:
        return f"No real positive WAVs found in {_project_real_positive_dir(project)}."
    valid_count = sum(1 for validation in validations if validation.ok)
    invalid_count = len(validations) - valid_count
    warning = ""
    if valid_count < project.real_positive_min_count:
        warning = (
            f" Fewer than {project.real_positive_min_count} real positives "
            "are present."
        )
    return (
        f"Real positives: {valid_count} valid, {invalid_count} invalid in "
        f"{_project_real_positive_dir(project)}. "
        f"Target training mix: {project.real_positive_target_percent}% real."
        f"{warning}"
    )


def _target_real_positive_count(
    synthetic_count: int,
    source_count: int,
    target_percent: int,
) -> int:
    """Return staged real copy count needed for the requested training mix."""
    if source_count <= 0:
        return 0
    if target_percent >= 100:
        return max(source_count, synthetic_count)
    if synthetic_count <= 0:
        return source_count
    percent = min(100, max(10, int(target_percent)))
    return max(
        source_count,
        (synthetic_count * percent + (100 - percent) - 1)
        // (100 - percent),
    )


def _project_real_positive_dir(project: WakeWordProject) -> Path:
    """Return the canonical project-local real positive directory."""
    return project.real_positives_dir


def _next_destination_path(directory: Path, stem: str) -> Path:
    """Return a non-conflicting managed destination path."""
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in stem
    ).strip("_") or "real_positive"
    index = 0
    while True:
        suffix = f"_{index:04d}" if index else ""
        candidate = directory / f"{safe_stem}{suffix}.wav"
        if not candidate.exists():
            return candidate
        index += 1


def _read_as_mono_int16(path: Path) -> np.ndarray:
    """Read a WAV as mono 16 kHz int16 audio."""
    sample_rate, data = wavfile.read(path)
    mono = _to_mono_float(data)
    if sample_rate != TARGET_SAMPLE_RATE:
        factor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TARGET_SAMPLE_RATE // factor,
            sample_rate // factor,
        )
    mono = np.clip(mono, -1.0, 1.0)
    return (mono * np.iinfo(np.int16).max).astype(np.int16)


def _normalize_wake_phrase_audio(audio: np.ndarray) -> np.ndarray:
    """Trim excessive edge silence while retaining short context."""
    if audio.size == 0:
        return audio.astype(np.int16)
    active = np.flatnonzero(np.abs(audio) > SILENCE_TRIM_THRESHOLD)
    if active.size == 0:
        return audio.astype(np.int16)
    keep = int(SILENCE_KEEP_SECONDS * TARGET_SAMPLE_RATE)
    start = max(int(active[0]) - keep, 0)
    end = min(int(active[-1]) + keep + 1, len(audio))
    return audio[start:end].astype(np.int16)


def _active_duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    """Return duration of non-silent content in seconds."""
    active = np.flatnonzero(np.abs(audio) > (SILENCE_TRIM_THRESHOLD / 32768.0))
    if active.size == 0:
        return 0.0
    return (int(active[-1]) - int(active[0]) + 1) / float(sample_rate)


def _to_mono_float(data: np.ndarray) -> np.ndarray:
    """Convert WAV data to mono float samples in approximately -1..1."""
    original_dtype = data.dtype
    if data.dtype.kind == "f":
        mono = np.clip(data.astype(np.float32), -1.0, 1.0)
    elif original_dtype == np.int16:
        mono = data.astype(np.float32) / np.iinfo(np.int16).max
    elif original_dtype == np.int32:
        mono = data.astype(np.float32) / np.iinfo(np.int32).max
    elif original_dtype == np.uint8:
        mono = (data.astype(np.float32) - 128.0) / 128.0
    else:
        mono = data.astype(np.float32)
        max_value = np.max(np.abs(mono)) or 1.0
        mono = mono / max_value
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    return mono


def _is_clipped(data: np.ndarray) -> bool:
    """Return whether a meaningful fraction of samples hits full scale."""
    if data.dtype.kind == "f":
        absolute = np.abs(data)
        return bool(np.mean(absolute >= 0.999) > CLIPPING_RATIO_LIMIT)
    if not np.issubdtype(data.dtype, np.integer):
        return False
    info = np.iinfo(data.dtype)
    clipped = (data <= info.min) | (data >= info.max)
    return bool(np.mean(clipped) > CLIPPING_RATIO_LIMIT)
