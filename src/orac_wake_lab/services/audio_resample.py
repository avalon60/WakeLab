"""Audio resampling helpers for Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-13
# Description: Normalises generated WAV clips to the rate openWakeWord expects.

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16000


def resample_wav_to_16khz(wav_path: Path) -> bool:
    """Resample a WAV file to 16 kHz in place.

    Args:
        wav_path (Path): WAV file path.

    Returns:
        bool: True if the file was resampled, otherwise False.
    """
    sample_rate, data = wavfile.read(wav_path)
    if sample_rate == TARGET_SAMPLE_RATE:
        return False

    if data.ndim > 1:
        data = data[:, 0]

    if data.dtype != np.int16:
        data = data.astype(np.int16)

    resampled = _resample_audio(data, sample_rate, TARGET_SAMPLE_RATE)
    wavfile.write(wav_path, TARGET_SAMPLE_RATE, resampled.astype(np.int16))
    return True


def resample_directory_to_16khz(directory: Path) -> int:
    """Resample every WAV in a directory to 16 kHz.

    Args:
        directory (Path): Directory containing WAV files.

    Returns:
        int: Number of files that were resampled.
    """
    count = 0
    for wav_path in sorted(directory.glob("*.wav")):
        if resample_wav_to_16khz(wav_path):
            count += 1
    return count


def _resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample 1-D PCM audio using polyphase filtering."""
    if source_rate == target_rate:
        return audio
    from math import gcd

    factor = gcd(source_rate, target_rate)
    up = target_rate // factor
    down = source_rate // factor
    return resample_poly(audio.astype(np.float32), up, down)
