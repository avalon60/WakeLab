"""Tests for the Orac Wake Lab home directory service."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Verifies managed path generation and auto-detection.

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from orac_wake_lab.services import wake_lab_home


def test_get_wake_lab_home_default() -> None:
    """It should default to ~/WakeLab."""
    with mock.patch.dict(os.environ, {}, clear=True):
        home = wake_lab_home.get_wake_lab_home()
        assert home == Path("~/WakeLab").expanduser().resolve()


def test_get_wake_lab_home_env_override() -> None:
    """It should respect WAKE_LAB_HOME."""
    custom_home = "/tmp/custom_wake_lab"
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": custom_home}):
        home = wake_lab_home.get_wake_lab_home()
        assert home == Path(custom_home).resolve()


def test_managed_paths_derive_from_home() -> None:
    """Conventional paths should be subdirectories of home."""
    home = wake_lab_home.get_wake_lab_home()
    assert wake_lab_home.get_projects_root() == home / "projects"
    assert wake_lab_home.get_data_dir() == home / "data"
    assert wake_lab_home.get_background_audio_dir() == home / "data" / "background"
    assert wake_lab_home.get_rir_dir() == home / "data" / "rir"
    assert wake_lab_home.get_negative_features_dir() == home / "data" / "features" / "negative"
    assert wake_lab_home.get_false_positive_validation_dir() == home / "data" / "features" / "validation"


def test_initialize_wake_lab_folders(tmp_path: Path) -> None:
    """It should create the conventional directory structure."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        dirs = wake_lab_home.initialize_wake_lab_folders()
        
        assert (tmp_path / "projects").is_dir()
        assert (tmp_path / "data" / "background").is_dir()
        assert (tmp_path / "data" / "rir").is_dir()
        assert (tmp_path / "data" / "features" / "negative").is_dir()
        assert (tmp_path / "data" / "features" / "validation").is_dir()
        assert (tmp_path / "external" / "openWakeWord").is_dir()
        assert (tmp_path / "external" / "piper-sample-generator").is_dir()
        assert (tmp_path / "downloads").is_dir()
        assert (tmp_path / "cache").is_dir()
        assert len(dirs) == 10


def test_discover_negative_features(tmp_path: Path) -> None:
    """It should find .npy files in the managed negative features dir."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        neg_dir = wake_lab_home.get_negative_features_dir()
        neg_dir.mkdir(parents=True)
        
        (neg_dir / "feat1.npy").write_text("data")
        (neg_dir / "feat2.npy").write_text("data")
        (neg_dir / "not_npy.txt").write_text("data")
        
        discovered = wake_lab_home.discover_negative_features()
        assert "feat1" in discovered
        assert "feat2" in discovered
        assert "not_npy" not in discovered
        assert len(discovered) == 2


def test_discover_validation_features(tmp_path: Path) -> None:
    """It should find .npy files in the managed validation features dir."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        val_dir = wake_lab_home.get_false_positive_validation_dir()
        val_dir.mkdir(parents=True)
        
        (val_dir / "val1.npy").write_text("data")
        
        discovered = wake_lab_home.discover_validation_features()
        assert len(discovered) == 1
        assert discovered[0].name == "val1.npy"


def test_detect_openwakeword_repo_fallback(tmp_path: Path) -> None:
    """It should fallback to managed external dir if not found elsewhere."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        with mock.patch("orac_wake_lab.services.wake_lab_home.Path.exists", autospec=True) as mock_exists:
            mock_exists.return_value = False
            # No repos exist anywhere
            detected = wake_lab_home.detect_openwakeword_repo()
            assert detected == tmp_path / "external" / "openWakeWord"


def test_detect_openwakeword_repo_discovery(tmp_path: Path) -> None:
    """It should detect openWakeWord if it exists in managed external."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        repo_dir = tmp_path / "external" / "openWakeWord"
        
        def side_effect(self):
            # Only return True for our mock repo's marker file
            return "openwakeword/train.py" in str(self) and str(repo_dir) in str(self)

        with mock.patch("orac_wake_lab.services.wake_lab_home.Path.exists", autospec=True) as mock_exists:
            mock_exists.side_effect = side_effect
            detected = wake_lab_home.detect_openwakeword_repo()
            assert detected == repo_dir.resolve()
