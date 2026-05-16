"""Managed home directory service for WakeLab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Centralises default paths and folder initialisation.

from __future__ import annotations

import os
from pathlib import Path


def get_wake_lab_home() -> Path:
    """Return the managed Wake Lab home directory.

    Returns:
        Path: Path to ~/WakeLab or WAKE_LAB_HOME.
    """
    env_home = os.environ.get("WAKE_LAB_HOME")
    if env_home:
        return Path(env_home).resolve()
    return Path("~/WakeLab").expanduser().resolve()



def get_projects_root() -> Path:
    """Return the default projects root."""
    return get_wake_lab_home() / "projects"


def get_data_dir() -> Path:
    """Return the data directory."""
    return get_wake_lab_home() / "data"


def get_background_audio_dir() -> Path:
    """Return the background audio directory."""
    return get_data_dir() / "background"


def get_rir_dir() -> Path:
    """Return the RIR directory."""
    return get_data_dir() / "rir"


def get_features_dir() -> Path:
    """Return the features directory."""
    return get_data_dir() / "features"


def get_negative_features_dir() -> Path:
    """Return the negative features directory."""
    return get_features_dir() / "negative"


def get_false_positive_validation_dir() -> Path:
    """Return the validation features directory."""
    return get_features_dir() / "validation"


def get_external_dir() -> Path:
    """Return the external tools directory."""
    return get_wake_lab_home() / "external"


def get_openwakeword_repo_dir() -> Path:
    """Return the managed openWakeWord checkout directory."""
    return get_external_dir() / "openWakeWord"


def get_piper_sample_generator_dir() -> Path:
    """Return the managed Piper sample generator checkout directory."""
    return get_external_dir() / "piper-sample-generator"


def get_piper_voices_dir() -> Path:
    """Return the managed Piper voice model directory."""
    return get_external_dir() / "piper-voices"


def get_orac_repo_dir() -> Path:
    """Return the managed runtime target directory."""
    return get_external_dir() / "runtime-target"


def get_downloads_dir() -> Path:
    """Return the downloads directory."""
    return get_wake_lab_home() / "downloads"


def get_cache_dir() -> Path:
    """Return the cache directory."""
    return get_wake_lab_home() / "cache"


def detect_openwakeword_repo() -> Path:
    """Return the managed openWakeWord repository path.

    Returns:
        Path: Managed default checkout path.
    """
    return get_openwakeword_repo_dir()


def detect_piper_sample_generator_path() -> Path:
    """Attempt to detect the Piper sample generator path.

    Returns:
        Path: Detected path or managed default.
    """
    candidates = [
        get_piper_sample_generator_dir(),
        Path("/home/clive/PycharmProjects/piper-sample-generator"),
        Path("~/PycharmProjects/piper-sample-generator").expanduser(),
    ]
    for candidate in candidates:
        if (candidate / "piper_sample_generator" / "__main__.py").exists():
            return candidate.resolve()
    return get_piper_sample_generator_dir()


def detect_orac_repo() -> Path:
    """Return the default runtime target path.

    Returns:
        Path: Managed default runtime target path.
    """
    return get_orac_repo_dir()


def initialize_wake_lab_folders() -> list[Path]:
    """Create the managed Wake Lab directory structure.

    Returns:
        list[Path]: List of created directories.
    """
    dirs = [
        get_projects_root(),
        get_background_audio_dir(),
        get_rir_dir(),
        get_negative_features_dir(),
        get_false_positive_validation_dir(),
        get_external_dir(),
        get_openwakeword_repo_dir(),
        get_piper_sample_generator_dir(),
        get_piper_voices_dir(),
        get_orac_repo_dir(),
        get_downloads_dir(),
        get_cache_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def discover_negative_features() -> dict[str, Path]:
    """Discover .npy files in the managed negative features directory.

    Returns:
        dict[str, Path]: Map of stem names to paths.
    """
    dir_path = get_negative_features_dir()
    if not dir_path.exists():
        return {}
    return {p.stem: p for p in dir_path.glob("*.npy")}


def discover_validation_features() -> list[Path]:
    """Discover .npy files in the managed validation features directory.

    Returns:
        list[Path]: List of .npy file paths.
    """
    dir_path = get_false_positive_validation_dir()
    if not dir_path.exists():
        return []
    return sorted(list(dir_path.glob("*.npy")))
