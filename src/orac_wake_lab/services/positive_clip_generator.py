"""Positive clip generation helpers with explicit phrase boundaries."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Generates phrase-part TTS clips and concatenates them with silence.

from __future__ import annotations

import random
import shutil
import tempfile
import wave
from collections.abc import Callable
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any


GenerateSamples = Callable[..., None]


def generate_samples_with_word_boundaries(
    *,
    base_generator: GenerateSamples,
    text: object,
    output_dir: str | Path,
    phrase_parts: list[str],
    silence_min_ms: int,
    silence_max_ms: int,
    max_samples: int | None = None,
    file_names: Iterable[str] | None = None,
    **kwargs: Any,
) -> None:
    """Generate positive samples from separate phrase parts.

    Args:
        base_generator (GenerateSamples): Piper-compatible sample generator.
        text (object): Original text argument, used only to infer sample count.
        output_dir (str | Path): Final output directory.
        phrase_parts (list[str]): Non-empty phrase fragments to generate.
        silence_min_ms (int): Minimum silence duration between parts.
        silence_max_ms (int): Maximum silence duration between parts.
        max_samples (int | None): Number of final clips to generate.
        file_names (Iterable[str] | None): Optional final WAV names.
        **kwargs (Any): Additional Piper generation keyword arguments.
    """
    clean_parts = [part.strip() for part in phrase_parts]
    if not clean_parts or any(not part for part in clean_parts):
        raise ValueError("Training phrase parts must not be empty.")
    if silence_min_ms < 0 or silence_max_ms < 0:
        raise ValueError("Inter-part silence values must be non-negative.")
    if silence_max_ms < silence_min_ms:
        raise ValueError(
            "Inter-part silence max must be greater than or equal to min."
        )

    sample_count = _resolve_sample_count(text, max_samples)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_names = _resolve_file_names(sample_count, file_names)

    temp_root = Path(
        tempfile.mkdtemp(prefix="wakelab_parts_", dir=str(output_path.parent))
    )
    try:
        part_dirs = _generate_part_clips(
            base_generator=base_generator,
            temp_root=temp_root,
            phrase_parts=clean_parts,
            final_names=final_names,
            sample_count=sample_count,
            **kwargs,
        )
        for file_name in final_names:
            part_paths = [part_dir / file_name for part_dir in part_dirs]
            concatenate_wavs_with_random_silence(
                part_paths,
                output_path / file_name,
                silence_min_ms=silence_min_ms,
                silence_max_ms=silence_max_ms,
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def concatenate_wavs_with_random_silence(
    wav_paths: list[Path],
    output_path: Path,
    *,
    silence_min_ms: int,
    silence_max_ms: int,
) -> list[int]:
    """Concatenate WAV files with random silence between each part.

    Args:
        wav_paths (list[Path]): Ordered WAV fragments.
        output_path (Path): Final concatenated WAV path.
        silence_min_ms (int): Minimum silence duration between fragments.
        silence_max_ms (int): Maximum silence duration between fragments.

    Returns:
        list[int]: Inserted silence durations in milliseconds.
    """
    if not wav_paths:
        raise ValueError("At least one WAV part is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inserted_durations: list[int] = []
    with wave.open(str(wav_paths[0]), "rb") as first_wav:
        params = first_wav.getparams()
        frames = [first_wav.readframes(first_wav.getnframes())]

    for wav_path in wav_paths[1:]:
        silence_ms = random.randint(silence_min_ms, silence_max_ms)
        inserted_durations.append(silence_ms)
        frames.append(_silence_frames(params, silence_ms))
        with wave.open(str(wav_path), "rb") as part_wav:
            if _wav_format(part_wav.getparams()) != _wav_format(params):
                raise ValueError(
                    "Generated phrase-part WAV files have incompatible formats."
                )
            frames.append(part_wav.readframes(part_wav.getnframes()))

    with wave.open(str(output_path), "wb") as output_wav:
        output_wav.setparams(params)
        output_wav.writeframes(b"".join(frames))
    return inserted_durations


def _generate_part_clips(
    *,
    base_generator: GenerateSamples,
    temp_root: Path,
    phrase_parts: list[str],
    final_names: list[str],
    sample_count: int,
    **kwargs: Any,
) -> list[Path]:
    """Generate each phrase part into a temporary clip directory."""
    part_dirs: list[Path] = []
    for index, part in enumerate(phrase_parts):
        part_dir = temp_root / f"part_{index}"
        part_dir.mkdir(parents=True, exist_ok=True)
        base_generator(
            text=[part],
            output_dir=part_dir,
            max_samples=sample_count,
            file_names=final_names,
            **kwargs,
        )
        part_dirs.append(part_dir)
    return part_dirs


def _resolve_sample_count(text: object, max_samples: int | None) -> int:
    """Return the number of final clips to generate."""
    if max_samples is not None:
        return max_samples
    if isinstance(text, list):
        return len(text)
    return 1


def _resolve_file_names(
    sample_count: int,
    file_names: Iterable[str] | None,
) -> list[str]:
    """Return deterministic final file names for the requested samples."""
    if file_names is None:
        return [f"{index}.wav" for index in range(sample_count)]
    return list(islice(file_names, sample_count))


def _silence_frames(params: wave._wave_params, silence_ms: int) -> bytes:
    """Return zeroed PCM frames for the requested WAV parameters."""
    frame_count = int(params.framerate * silence_ms / 1000)
    return b"\x00" * frame_count * params.nchannels * params.sampwidth


def _wav_format(
    params: wave._wave_params,
) -> tuple[int, int, int, str, str]:
    """Return the WAV fields that must match for concatenation."""
    return (
        params.nchannels,
        params.sampwidth,
        params.framerate,
        params.comptype,
        params.compname,
    )
