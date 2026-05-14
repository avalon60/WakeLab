"""Test trained openWakeWord ONNX models against WAV files."""
# Author: Clive Bostock
# Date: 2026-05-13
# Description: Provides file-based inference for trained wake-word models.

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.services.openwakeword_assets import (
    ensure_openwakeword_training_assets,
)


TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class WavTestResult:
    """Result from testing a WAV file against a wake-word model."""

    model_path: Path
    wav_path: Path
    model_name: str
    threshold: float
    max_score: float
    frame_count: int

    @property
    def activated(self) -> bool:
        """Return whether the highest frame score meets the threshold."""
        return self.max_score >= self.threshold

    def render(self) -> str:
        """Return a human-readable diagnostic summary."""
        status = "ACTIVATED" if self.activated else "not activated"
        return (
            f"Model: {self.model_path}\n"
            f"WAV: {self.wav_path}\n"
            f"Model output: {self.model_name}\n"
            f"Frames evaluated: {self.frame_count}\n"
            f"Max score: {self.max_score:.4f}\n"
            f"Threshold: {self.threshold:.4f}\n"
            f"Result: {status}\n"
        )


def default_model_path(project: WakeWordProject) -> Path:
    """Return the expected ONNX model path for a project."""
    return project.openwakeword_output_dir / f"{project.model_name}.onnx"


def test_wav_file(
    project: WakeWordProject,
    model_path: Path,
    wav_path: Path,
    threshold: float,
) -> WavTestResult:
    """Run a WAV file through a trained openWakeWord ONNX model.

    Args:
        project: Project containing the openWakeWord repository path.
        model_path: Trained ONNX model path.
        wav_path: WAV file to test.
        threshold: Activation threshold used for the summary.

    Returns:
        WavTestResult: Aggregated prediction result.

    Raises:
        FileNotFoundError: If the model or WAV file is missing.
        ValueError: If the threshold is outside the supported range.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file does not exist: {wav_path}")
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1.")

    ensure_openwakeword_training_assets(project.openwakeword_repo)
    samples = _read_wav_as_16khz_mono_int16(wav_path)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(project.openwakeword_repo))
        import openwakeword

        model = openwakeword.Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
        )
        predictions = model.predict_clip(samples)
    finally:
        sys.path = original_path

    model_name, max_score, frame_count = summarize_predictions(predictions)
    return WavTestResult(
        model_path=model_path,
        wav_path=wav_path,
        model_name=model_name,
        threshold=threshold,
        max_score=max_score,
        frame_count=frame_count,
    )


def summarize_predictions(
    predictions: list[dict[str, float]],
) -> tuple[str, float, int]:
    """Return the highest-scoring model label and score."""
    max_model = ""
    max_score = 0.0
    for frame in predictions:
        for model_name, score in frame.items():
            score_value = float(score)
            if score_value > max_score:
                max_model = model_name
                max_score = score_value
    return max_model, max_score, len(predictions)


def _read_wav_as_16khz_mono_int16(path: Path) -> np.ndarray:
    """Read a WAV file and convert it to 16 kHz mono int16 samples."""
    sample_rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype.kind == "f":
        data = np.clip(data, -1.0, 1.0)
        data = (data * np.iinfo(np.int16).max).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.float32)
        max_value = np.max(np.abs(data)) or 1
        data = (data / max_value * np.iinfo(np.int16).max).astype(np.int16)
    if sample_rate != TARGET_SAMPLE_RATE:
        gcd = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        data = signal.resample_poly(
            data,
            TARGET_SAMPLE_RATE // gcd,
            sample_rate // gcd,
        ).astype(np.int16)
    return data
