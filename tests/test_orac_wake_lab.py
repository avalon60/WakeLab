"""Tests for the Orac Wake Lab Phase 1 services."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Covers wake-word project validation and config generation.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from orac_wake_lab.models.training_job import JobStatus
from orac_wake_lab.models.training_job import TrainingJob
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import derive_model_name
from orac_wake_lab.models.validation import ValidationResult
from orac_wake_lab.models.validation import validate_model_name
from orac_wake_lab.models.validation import validate_phrase
from orac_wake_lab.services.dependency_checks import run_dependency_checks
from orac_wake_lab.services.orac_export import build_orac_config_snippet
from orac_wake_lab.services.orac_export import build_smoke_test_command
from orac_wake_lab.services.orac_export import export_model_to_orac
from orac_wake_lab.services.process_runner import ProcessRunner
from orac_wake_lab.services.project_store import create_project_workspace
from orac_wake_lab.services.project_store import load_project
from orac_wake_lab.services.training_config import write_training_config
from orac_wake_lab.services.training_config import (
    validate_training_config_inputs,
)


def test_derive_model_name_from_phrase() -> None:
    """Model names should be derived from wake phrases."""
    assert derive_model_name("Hey Orac") == "hey_orac"
    assert derive_model_name("  Hey, ORAC!  ") == "hey_orac"


def test_validation_rejects_invalid_model_name() -> None:
    """Model name validation should enforce the supported pattern."""
    assert validate_model_name("hey_orac").status == "pass"
    assert validate_model_name("Hey-Orac").status == "fail"
    assert validate_phrase("").status == "fail"
    assert validate_phrase("Orac").status == "warn"


def test_project_workspace_creation(tmp_path: Path) -> None:
    """Project creation should create all Phase 1 directories."""
    project = _project(tmp_path)
    create_project_workspace(project)

    expected = [
        "project.json",
        "config",
        "openwakeword_output",
        "models",
        "logs",
        "export",
        "test/activations",
        "test/false_positives",
        "test/false_negatives",
    ]
    for relative_path in expected:
        assert (project.workspace_dir / relative_path).exists()


def test_training_config_generation(tmp_path: Path) -> None:
    """Generated YAML should contain openWakeWord training keys."""
    project = _project(tmp_path)
    create_project_workspace(project)
    config_path = write_training_config(project)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["target_phrase"] == ["Hey Orac"]
    assert config["model_name"] == "hey_orac"
    assert config["output_dir"] == str(project.openwakeword_output_dir)
    assert config["piper_sample_generator_path"] == "/tmp/piper"
    assert config["background_paths"] == ["/tmp/background"]
    assert config["rir_paths"] == ["/tmp/rir"]
    assert config["false_positive_validation_data_path"] == "/tmp/fp.npy"
    assert config["feature_data_files"] == {
        "negative_features": "/tmp/negative_features.npy"
    }
    assert "false_positive_sample" not in config["feature_data_files"]
    assert config["model_type"] == "dnn"
    assert "steps" in config


def test_training_config_validation_requires_negative_features(
    tmp_path: Path,
) -> None:
    """Training validation should fail without negative feature files."""
    project = _project(tmp_path)
    project.negative_feature_data_files = {}

    result = validate_training_config_inputs(project)

    assert result.status == "fail"
    assert "train" in result.blocks


def test_orac_config_snippet_generation() -> None:
    """Orac config snippets should point at the runtime model location."""
    snippet = build_orac_config_snippet(Path("hey_orac.tflite"))

    assert "activation_mode = openwakeword" in snippet
    assert (
        "openwakeword_model_paths = "
        "${ORAC_HOME}/var/models/wake/hey_orac.tflite"
    ) in snippet
    assert "openwakeword_model_names =" in snippet


def test_export_refuses_overwrite_by_default(tmp_path: Path) -> None:
    """Export should not replace an existing model unless requested."""
    project = _project(tmp_path)
    model_file = tmp_path / "hey_orac.tflite"
    model_file.write_text("new", encoding="utf-8")
    target_dir = project.orac_repo / "var" / "models" / "wake"
    target_dir.mkdir(parents=True)
    (target_dir / model_file.name).write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_model_to_orac(project, model_file)

    assert (target_dir / model_file.name).read_text(encoding="utf-8") == "old"


def test_export_allows_explicit_overwrite(tmp_path: Path) -> None:
    """Export should replace an existing model only when requested."""
    project = _project(tmp_path)
    model_file = tmp_path / "hey_orac.tflite"
    model_file.write_text("new", encoding="utf-8")
    target_dir = project.orac_repo / "var" / "models" / "wake"
    target_dir.mkdir(parents=True)
    (target_dir / model_file.name).write_text("old", encoding="utf-8")

    target, _config_path, _smoke = export_model_to_orac(
        project,
        model_file,
        overwrite=True,
    )

    assert target.read_text(encoding="utf-8") == "new"


def test_smoke_test_command_uses_project_orac_repo(tmp_path: Path) -> None:
    """Smoke-test command should use the configured Orac repo path."""
    project = _project(tmp_path)

    assert str(project.orac_repo) in build_smoke_test_command(project)


def test_project_load_save_round_trip(tmp_path: Path) -> None:
    """Project metadata should round-trip through project.json."""
    project = _project(tmp_path)
    create_project_workspace(project)

    loaded = load_project(project.workspace_dir / "project.json")

    assert loaded.model_name == project.model_name
    assert loaded.negative_feature_data_files == (
        project.negative_feature_data_files
    )


def test_dependency_check_results_are_structured(tmp_path: Path) -> None:
    """Dependency checks should return structured statuses."""
    results = run_dependency_checks(_project(tmp_path))

    assert results
    assert all(isinstance(result, ValidationResult) for result in results)
    assert {result.status for result in results}.issubset(
        {"pass", "warn", "fail"}
    )


def test_process_runner_prevents_concurrent_jobs(tmp_path: Path) -> None:
    """Runner should reject a second job while one is running."""
    runner = ProcessRunner(cancel_timeout_seconds=0.2)
    first = _sleep_job(tmp_path, "first")
    second = _sleep_job(tmp_path, "second")
    runner.start(first)
    _wait_until(lambda: runner.is_running)

    with pytest.raises(RuntimeError):
        runner.start(second)

    runner.cancel()
    _wait_until(lambda: first.status == JobStatus.CANCELLED, timeout=3.0)


def test_process_runner_cancel_status_after_exit(tmp_path: Path) -> None:
    """Cancellation should not mark CANCELLED before process exit."""
    runner = ProcessRunner(cancel_timeout_seconds=0.2)
    job = _sigterm_ignoring_job(tmp_path)
    runner.start(job)
    _wait_until(lambda: runner.is_running)
    _wait_until(lambda: "ready" in job.log_path.read_text(encoding="utf-8"))

    runner.cancel()

    assert job.status == JobStatus.RUNNING
    _wait_until(lambda: job.status == JobStatus.CANCELLED, timeout=3.0)


def _project(tmp_path: Path) -> WakeWordProject:
    return WakeWordProject(
        wake_phrase="Hey Orac",
        model_name="hey_orac",
        workspace_dir=tmp_path / "hey_orac",
        piper_sample_generator_path=Path("/tmp/piper"),
        background_paths=[Path("/tmp/background")],
        rir_paths=[Path("/tmp/rir")],
        negative_feature_data_files={
            "negative_features": Path("/tmp/negative_features.npy")
        },
        false_positive_validation_data_path=Path("/tmp/fp.npy"),
        custom_negative_phrases=["hey oracle"],
        profile="quick",
        orac_repo=tmp_path / "orac",
    )


def _sleep_job(tmp_path: Path, name: str) -> TrainingJob:
    return TrainingJob(
        name=name,
        command=[
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
        cwd=tmp_path,
        log_path=tmp_path / f"{name}.log",
    )


def _sigterm_ignoring_job(tmp_path: Path) -> TrainingJob:
    return TrainingJob(
        name="ignore_sigterm",
        command=[
            sys.executable,
            "-c",
            (
                "import signal, time; "
                "signal.signal(signal.SIGTERM, lambda *_: None); "
                "print('ready', flush=True); "
                "time.sleep(10)"
            ),
        ],
        cwd=tmp_path,
        log_path=tmp_path / "ignore_sigterm.log",
    )


def _wait_until(condition: object, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for condition.")
