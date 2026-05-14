"""Tests for the managed openWakeWord feature bundle workflow."""
# Author: Clive Bostock
# Date: 2026-05-10
# Description: Covers feature bundle detection, registration, and config use.

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.services.feature_bundle import (
    STANDARD_NEGATIVE_FEATURE_FILENAME,
)
from orac_wake_lab.services.feature_bundle import (
    STANDARD_VALIDATION_FEATURE_FILENAME,
)
from orac_wake_lab.services.feature_bundle import (
    apply_standard_feature_bundle_to_project,
)
from orac_wake_lab.services.feature_bundle import (
    detect_standard_feature_bundle,
)
from orac_wake_lab.services.feature_bundle import (
    standard_feature_bundle_paths,
)
from orac_wake_lab.services.feature_bundle import (
    download_standard_feature_bundle,
)
from orac_wake_lab.services.feature_bundle import (
    validate_feature_file,
)
from orac_wake_lab.services.training_config import build_training_config


def test_download_progress_callback(tmp_path: Path) -> None:
    """The download should report progress through the callback."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        # Mocking urlopen to avoid real network calls
        mock_response = mock.MagicMock()
        mock_response.info.return_value.get.return_value = "100"  # 100 bytes
        # We need 3 calls per file (2 data, 1 empty) -> 6 calls total
        mock_response.read.side_effect = [
            b"a" * 50, b"b" * 50, b"",
            b"c" * 50, b"d" * 50, b""
        ]
        mock_response.__enter__.return_value = mock_response

        progress_calls = []

        def callback(filename: str, downloaded: int, total: int) -> None:
            progress_calls.append((filename, downloaded, total))

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            # We also need to mock np.load because download calls detect_standard_feature_bundle
            with mock.patch("numpy.load"):
                download_standard_feature_bundle(progress_callback=callback)

        # We expect 2 progress calls for each of the 2 files = 4 calls total.
        assert len(progress_calls) == 4
        # Check first file progress
        assert progress_calls[0][1] == 50
        assert progress_calls[0][2] == 100
        assert progress_calls[1][1] == 100
        assert progress_calls[1][2] == 100


def test_detects_complete_managed_feature_bundle(tmp_path: Path) -> None:
    """Managed feature bundle files should be detected when present."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        paths = standard_feature_bundle_paths()
        paths.negative.parent.mkdir(parents=True, exist_ok=True)
        paths.validation.parent.mkdir(parents=True, exist_ok=True)
        np.save(paths.negative, np.array([1.0, 2.0], dtype=np.float32))
        np.save(paths.validation, np.array([3.0, 4.0], dtype=np.float32))

        bundle = detect_standard_feature_bundle()

        assert bundle.is_ready
        assert bundle.negative.status == "pass"
        assert bundle.validation.status == "pass"
        assert bundle.paths.negative.name == STANDARD_NEGATIVE_FEATURE_FILENAME
        assert (
            bundle.paths.validation.name
            == STANDARD_VALIDATION_FEATURE_FILENAME
        )


def test_reports_missing_negative_feature_file(tmp_path: Path) -> None:
    """Missing negative bundle files should be reported plainly."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        paths = standard_feature_bundle_paths()
        paths.validation.parent.mkdir(parents=True, exist_ok=True)
        np.save(paths.validation, np.array([1.0], dtype=np.float32))

        bundle = detect_standard_feature_bundle()

        assert bundle.is_ready is False
        assert bundle.negative.status == "fail"
        assert "cannot find" in bundle.negative.message.lower()
        assert STANDARD_NEGATIVE_FEATURE_FILENAME in bundle.negative.message


def test_reports_missing_validation_feature_file(tmp_path: Path) -> None:
    """Missing validation bundle files should be reported plainly."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        paths = standard_feature_bundle_paths()
        paths.negative.parent.mkdir(parents=True, exist_ok=True)
        np.save(paths.negative, np.array([1.0], dtype=np.float32))

        bundle = detect_standard_feature_bundle()

        assert bundle.is_ready is False
        assert bundle.validation.status == "fail"
        assert "cannot find" in bundle.validation.message.lower()
        assert STANDARD_VALIDATION_FEATURE_FILENAME in bundle.validation.message


def test_rejects_non_npy_paths(tmp_path: Path) -> None:
    """Feature validation should reject non-.npy files."""
    bad_path = tmp_path / "feature.txt"
    bad_path.write_text("not a feature file", encoding="utf-8")

    result = validate_feature_file(
        bad_path,
        "Negative/background feature file",
    )

    assert result.status == "fail"
    assert ".npy" in result.message


def test_preserves_manual_advanced_feature_paths(tmp_path: Path) -> None:
    """Managed bundle detection should not overwrite advanced manual paths."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        paths = standard_feature_bundle_paths()
        paths.negative.parent.mkdir(parents=True, exist_ok=True)
        paths.validation.parent.mkdir(parents=True, exist_ok=True)
        np.save(paths.negative, np.array([1.0], dtype=np.float32))
        np.save(paths.validation, np.array([2.0], dtype=np.float32))

        project = _project(tmp_path)
        manual_negative = tmp_path / "advanced" / "negative.npy"
        manual_validation = tmp_path / "advanced" / "validation.npy"
        manual_negative.parent.mkdir(parents=True, exist_ok=True)
        np.save(manual_negative, np.array([5.0], dtype=np.float32))
        np.save(manual_validation, np.array([6.0], dtype=np.float32))
        project.negative_feature_data_files = {
            "manual_negative": manual_negative,
        }
        project.false_positive_validation_data_path = manual_validation

        updated = apply_standard_feature_bundle_to_project(
            project,
            force=False,
        )

        assert updated is False
        assert project.negative_feature_data_files == {
            "manual_negative": manual_negative,
        }
        assert project.false_positive_validation_data_path == manual_validation


def test_generated_training_config_contains_resolved_feature_paths(
    tmp_path: Path,
) -> None:
    """Generated training config should point at the resolved feature bundle."""
    with mock.patch.dict(os.environ, {"WAKE_LAB_HOME": str(tmp_path)}):
        paths = standard_feature_bundle_paths()
        paths.negative.parent.mkdir(parents=True, exist_ok=True)
        paths.validation.parent.mkdir(parents=True, exist_ok=True)
        np.save(paths.negative, np.array([1.0], dtype=np.float32))
        np.save(paths.validation, np.array([2.0], dtype=np.float32))

        project = _project(tmp_path)
        apply_standard_feature_bundle_to_project(project, force=True)

        config = build_training_config(project)

        assert config["feature_data_files"] == {
            "negative_features": str(paths.negative)
        }
        assert config["false_positive_validation_data_path"] == str(
            paths.validation
        )


def _project(tmp_path: Path) -> WakeWordProject:
    """Build a project fixture for feature bundle tests."""
    return WakeWordProject(
        wake_phrase="Hey Orac",
        model_name="hey_orac",
        workspace_dir=tmp_path / "hey_orac",
        openwakeword_repo=tmp_path / "openwakeword",
        piper_sample_generator_path=tmp_path / "piper",
        background_paths=[Path("/tmp/background")],
        rir_paths=[Path("/tmp/rir")],
        negative_feature_data_files={},
        false_positive_validation_data_path=Path(""),
        custom_negative_phrases=["hey oracle"],
        profile="quick",
        orac_repo=tmp_path / "orac",
    )
