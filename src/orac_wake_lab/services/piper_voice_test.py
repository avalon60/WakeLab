"""Piper voice test playback helpers for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-13
# Description: Synthesises and plays a short voice sample for Project tab testing.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


VOICE_TEST_TEXT = (
    "I've seen things you people wouldn't believe. "
    "Attack ships on fire off the shoulder of Orion. "
    "I watched C-Beams glitter in the dark. "
    "Near the Tannhäuser Gates. "
    "All those moments will be lost in time, like tears in rain. "
    "Time to die."
)

__test__ = False


def test_piper_voice(model_path: Path, generator_path: Path) -> None:
    """Synthesise and play a short sample with a Piper voice model.

    Args:
        model_path (Path): Piper voice or generator model to test.
        generator_path (Path): Local ``piper-sample-generator`` checkout.

    Raises:
        FileNotFoundError: If required files or directories are missing.
        ValueError: If the model path is empty or uses an unsupported suffix.
        RuntimeError: If synthesis or playback fails.
    """
    if not str(model_path).strip() or str(model_path).strip() in {"", "."}:
        raise ValueError("No Piper voice or generator model is configured.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Piper voice or generator model: {model_path}")
    if model_path.suffix not in {".onnx", ".pt"}:
        raise ValueError(
            "Piper voice or generator model must be a .onnx or .pt file: "
            f"{model_path}"
        )
    if not generator_path.exists():
        raise FileNotFoundError(
            f"Missing Piper sample generator repository: {generator_path}"
        )

    with tempfile.TemporaryDirectory(prefix="wakelab-voice-test-") as temp_dir:
        output_dir = Path(temp_dir)
        _synthesise_sample(
            model_path=model_path,
            generator_path=generator_path,
            output_dir=output_dir,
        )
        wav_path = _find_first_wav(output_dir)
        if wav_path is None:
            raise RuntimeError("Piper sample generator did not produce audio.")
        _play_wav(wav_path)


def _synthesise_sample(
    *,
    model_path: Path,
    generator_path: Path,
    output_dir: Path,
) -> None:
    """Run piper-sample-generator for a single sample."""
    env = dict(os.environ)
    python_path = env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    if python_path:
        env["PYTHONPATH"] = f"{generator_path}:{python_path}"
    else:
        env["PYTHONPATH"] = str(generator_path)

    command = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        VOICE_TEST_TEXT,
        "--model",
        str(model_path),
        "--max-samples",
        "1",
        "--batch-size",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(
        command,
        cwd=generator_path,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _find_first_wav(output_dir: Path) -> Path | None:
    """Return the first WAV produced by the generator."""
    wav_files = sorted(output_dir.glob("*.wav"))
    return wav_files[0] if wav_files else None


def _play_wav(wav_path: Path) -> None:
    """Play a WAV file using a local desktop audio player."""
    players = [
        ["aplay", str(wav_path)],
        ["paplay", str(wav_path)],
        ["pw-play", str(wav_path)],
        [
            "ffplay",
            "-autoexit",
            "-nodisp",
            "-loglevel",
            "error",
            str(wav_path),
        ],
    ]
    for command in players:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError("No supported audio playback command was found.")
