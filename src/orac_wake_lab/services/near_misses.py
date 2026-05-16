"""Near-miss evaluation clip management."""
# Author: Clive Bostock
# Date: 2026-05-16
# Description: Imports, validates, and synthesises near-miss evaluation clips.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scipy.io import wavfile

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.services.real_positives import (
    _normalize_wake_phrase_audio,
)
from orac_wake_lab.services.real_positives import (
    _read_as_mono_int16,
)
from orac_wake_lab.services.real_positives import (
    validate_real_positive_clip,
)


def import_near_miss_wavs(
    project: WakeWordProject,
    sources: list[Path],
) -> list[Path]:
    """Import WAV files into the project-managed near-misses directory."""
    destination_dir = project.near_miss_clips_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    imported: list[Path] = []
    for source in sources:
        validation = validate_real_positive_clip(source)
        if not validation.ok:
            raise ValueError(
                f"Invalid near-miss clip {source}: " + "; ".join(validation.messages)
            )
        destination = _next_destination_path(destination_dir, source.stem)
        audio = _read_as_mono_int16(source)
        normalized = _normalize_wake_phrase_audio(audio)
        wavfile.write(destination, 16000, normalized)
        imported.append(destination)
    return imported


def generate_synthetic_near_miss_wavs(project: WakeWordProject) -> list[Path]:
    """Generate near-miss WAVs from the project's negative phrases."""
    phrases = [phrase.strip() for phrase in project.custom_negative_phrases or []]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return []
    if _path_is_unset(project.piper_sample_generator_path):
        raise ValueError("Select a Piper sample generator first.")
    if _path_is_unset(project.piper_voice_model_path):
        raise ValueError("Select a Piper voice or generator model first.")
    if not project.piper_sample_generator_path.exists():
        raise FileNotFoundError(
            f"Missing Piper sample generator repository: {project.piper_sample_generator_path}"
        )
    if not project.piper_voice_model_path.exists():
        raise FileNotFoundError(
            f"Missing Piper voice or generator model: {project.piper_voice_model_path}"
        )

    destination_dir = project.near_miss_clips_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    for existing in destination_dir.glob("synthetic_near_miss_*.wav"):
        existing.unlink()

    generated_paths: list[Path] = []
    for index, phrase in enumerate(phrases):
        temp_dir = Path(tempfile.mkdtemp(prefix="wakelab-near-miss-"))
        try:
            _synthesise_phrase(phrase, project, temp_dir)
            wav_path = _first_wav(temp_dir)
            if wav_path is None:
                raise RuntimeError(
                    f"Piper sample generator did not produce audio for: {phrase}"
                )
            audio = _read_as_mono_int16(wav_path)
            normalized = _normalize_wake_phrase_audio(audio)
            destination = destination_dir / (
                f"synthetic_near_miss_{index:04d}_{_safe_stem(phrase)}.wav"
            )
            wavfile.write(destination, 16000, normalized)
            generated_paths.append(destination)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return generated_paths


def _synthesise_phrase(phrase: str, project: WakeWordProject, output_dir: Path) -> None:
    """Run piper-sample-generator for a single phrase into a temp directory."""
    env = dict(os.environ)
    python_path = env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    if python_path:
        env["PYTHONPATH"] = (
            f"{project.piper_sample_generator_path}:{python_path}"
        )
    else:
        env["PYTHONPATH"] = str(project.piper_sample_generator_path)
    env["WAKELAB_PIPER_MODEL_PATH"] = str(project.piper_voice_model_path)

    command = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        phrase,
        "--model",
        str(project.piper_voice_model_path),
        "--max-samples",
        "1",
        "--batch-size",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(
        command,
        cwd=project.piper_sample_generator_path,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _first_wav(directory: Path) -> Path | None:
    """Return the first WAV in a directory, if one exists."""
    wav_files = sorted(directory.glob("*.wav"))
    return wav_files[0] if wav_files else None


def _next_destination_path(directory: Path, stem: str) -> Path:
    """Return a non-conflicting destination WAV path."""
    safe_stem = _safe_stem(stem)
    index = 0
    while True:
        suffix = f"_{index:04d}" if index else ""
        candidate = directory / f"{safe_stem}{suffix}.wav"
        if not candidate.exists():
            return candidate
        index += 1


def _safe_stem(text: str) -> str:
    """Return a filesystem-safe stem for a generated clip."""
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in text
    ).strip("_") or "near_miss"


def _path_is_unset(path: object) -> bool:
    """Return whether a path-like field is intentionally blank."""
    return str(path).strip() in {"", "."}
