"""Tests for the Orac Wake Lab Phase 1 services."""
# Author: Clive Bostock
# Date: 2026-05-10
# Description: Covers wake-word project validation and config generation.

from __future__ import annotations

import json
import argparse
from collections import deque
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from orac_wake_lab.models.training_job import JobStatus
from orac_wake_lab.models.training_job import TrainingJob
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import derive_model_name
from orac_wake_lab.models.validation import ValidationResult
from orac_wake_lab.models.validation import validate_model_name
from orac_wake_lab.models.validation import validate_phrase
from orac_wake_lab.services.dependency_checks import run_dependency_checks
from orac_wake_lab.services import dependency_checks
from orac_wake_lab.services.audio_resample import resample_directory_to_16khz
from orac_wake_lab.services import model_tester
from orac_wake_lab.services import orac_export
from orac_wake_lab.services import near_misses
from orac_wake_lab.services.orac_export import build_orac_config_snippet
from orac_wake_lab.services.orac_export import build_smoke_test_command
from orac_wake_lab.services.orac_export import export_model_to_orac
from orac_wake_lab.services.orac_export import export_model_to_directory
from orac_wake_lab.services.orac_export import find_generated_models
from orac_wake_lab.services.process_runner import ProcessRunner
from orac_wake_lab.services import openwakeword_train_runner
from orac_wake_lab.services.project_store import create_project_workspace
from orac_wake_lab.services.project_store import delete_project_workspace
from orac_wake_lab.services.project_store import discover_projects
from orac_wake_lab.services.project_store import load_project
from orac_wake_lab.services.project_store import project_dir_for_model
from orac_wake_lab.services import piper_voice_test
from orac_wake_lab.services import positive_clip_generator
from orac_wake_lab.services import real_positives
from orac_wake_lab.services.training_config import build_training_config
from orac_wake_lab.services.training_config import write_training_config
from orac_wake_lab.services.training_config import (
    validate_positive_generation_settings,
)
from orac_wake_lab.services.training_config import validate_training_text_fields
from orac_wake_lab.services.training_config import (
    validate_training_config_inputs,
)
from orac_wake_lab.ui import main_window
from orac_wake_lab.ui import export_tab
from orac_wake_lab.ui import project_tab
from orac_wake_lab.ui import test_tab
from orac_wake_lab.ui import training_tab
from orac_wake_lab.ui.main_window import DEFAULT_APPEARANCE_MODE
from orac_wake_lab.ui.main_window import THEME_PATH
from orac_wake_lab.ui.main_window import OracWakeLabApp
from orac_wake_lab.ui.checks_tab import ChecksTab
from orac_wake_lab.ui.export_tab import ExportTab
from orac_wake_lab.ui.project_tab import ProjectTab
from orac_wake_lab.ui.project_tab import project_form_values
from orac_wake_lab.ui.training_tab import build_training_job


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
        "test/near_misses",
    ]
    for relative_path in expected:
        assert (project.workspace_dir / relative_path).exists()


def test_create_project_workspace_initialises_near_miss_directory(
    tmp_path: Path,
) -> None:
    """Project workspaces should always create the near-miss folder."""
    project = _project(tmp_path)
    create_project_workspace(project)

    assert project.near_miss_clips_dir.exists()
    assert project.near_miss_clips_dir.is_dir()


def test_training_config_generation(tmp_path: Path) -> None:
    """Generated YAML should contain openWakeWord training keys."""
    project = _project(tmp_path)
    create_project_workspace(project)
    config_path = write_training_config(project)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["target_phrase"] == ["Hey Orac"]
    assert config["model_name"] == "hey_orac"
    assert config["output_dir"] == str(project.openwakeword_output_dir)
    assert config["piper_sample_generator_path"] == str(project.piper_sample_generator_path)
    assert config["background_paths"] == [str(p) for p in project.background_paths]
    assert config["rir_paths"] == [str(p) for p in project.rir_paths]
    assert config["false_positive_validation_data_path"] == str(project.false_positive_validation_data_path)
    assert config["feature_data_files"] == {
        "negative_features": str(project.negative_feature_data_files["negative_features"])
    }
    assert "false_positive_sample" not in config["feature_data_files"]
    assert config["model_type"] == "dnn"
    assert "steps" in config


def test_training_config_keeps_canonical_model_name_with_phrase_parts(
    tmp_path: Path,
) -> None:
    """Pronunciation controls should not rename the exported model."""
    project = _project(tmp_path)
    project.training_pronunciation_phrase = "Hay O-rack"
    project.training_phrase_parts = ["Hey", "Orac"]

    config = build_training_config(project)

    assert config["model_name"] == "hey_orac"
    assert config["target_phrase"] == ["Hey Orac"]
    assert config["wakelab_positive_generation"] == {
        "training_pronunciation_phrase": "Hay O-rack",
        "training_phrase_parts": ["Hey", "Orac"],
        "inter_part_silence_min_ms": 80,
        "inter_part_silence_max_ms": 250,
    }
    assert config["wakelab_real_positives"] == {
        "enabled": True,
        "directory": str(project.real_positive_clips_dir),
        "minimum_count": 20,
        "target_percent": 30,
    }


def test_training_config_validation_requires_negative_features(
    tmp_path: Path,
) -> None:
    """Training validation should fail without negative feature files."""
    project = _project(tmp_path)
    project.negative_feature_data_files = {}

    result = validate_training_config_inputs(project)

    assert result.status == "fail"
    assert "train" in result.blocks


def test_positive_generation_validation_rejects_invalid_silence_bounds(
    tmp_path: Path,
) -> None:
    """Positive generation validation should reject invalid silence bounds."""
    project = _project(tmp_path)
    project.training_phrase_parts = ["Hey", "Orac"]
    project.inter_part_silence_min_ms = 250
    project.inter_part_silence_max_ms = 80

    result = validate_positive_generation_settings(project)

    assert result.status == "fail"
    assert "generate" in result.blocks


def test_positive_generation_validation_rejects_pipe_in_pronunciation_phrase(
    tmp_path: Path,
) -> None:
    """The separator should only be accepted in training phrase parts."""
    project = _project(tmp_path)
    project.training_pronunciation_phrase = "Hay | Aurack"

    result = validate_positive_generation_settings(project)

    assert result.status == "fail"
    assert "Training pronunciation phrase" in result.message
    assert "generate" in result.blocks


def test_training_text_validation_rejects_pipe_in_canonical_fields(
    tmp_path: Path,
) -> None:
    """The phrase-part separator should not reach openWakeWord text fields."""
    project = _project(tmp_path)
    project.wake_phrase = "Hey | Aurack"

    result = validate_training_text_fields(project)

    assert result.status == "fail"
    assert "Wake phrase" in result.message
    assert "generate" in result.blocks


def test_generate_clips_blocks_when_wake_phrase_contains_pipe(
    tmp_path: Path,
) -> None:
    """Generate Clips should fail early when a saved project has bad text."""
    project = _project(tmp_path)
    project.wake_phrase = "Hey | Aurack"

    errors = training_tab._blocking_errors_for_stage(project, "generate_clips")

    assert any("Do not put '|'" in message for message in errors)


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


def test_export_copies_onnx_external_data_sidecar(tmp_path: Path) -> None:
    """ONNX external-data sidecars must be exported with the model."""
    project = _project(tmp_path)
    model_file = tmp_path / "hey_orac.onnx"
    sidecar_file = tmp_path / "hey_orac.onnx.data"
    model_file.write_text("model", encoding="utf-8")
    sidecar_file.write_text("weights", encoding="utf-8")

    target, _config_path, _smoke = export_model_to_orac(project, model_file)

    assert target.read_text(encoding="utf-8") == "model"
    assert (target.parent / "hey_orac.onnx.data").read_text(
        encoding="utf-8"
    ) == "weights"


def test_export_embeds_onnx_external_data_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONNX exports should be able to collapse external data into one file."""
    model_file = tmp_path / "hey_orac.onnx"
    model_file.write_text("model", encoding="utf-8")
    target_dir = tmp_path / "export"
    called: dict[str, Path] = {}

    def fake_embed(source_onnx: Path, target_onnx: Path) -> None:
        called["source"] = source_onnx
        called["target"] = target_onnx
        target_onnx.write_text("embedded", encoding="utf-8")

    monkeypatch.setattr(
        orac_export,
        "embed_onnx_external_data_file",
        fake_embed,
    )

    target = export_model_to_directory(
        model_file,
        target_dir,
        embed_onnx_external_data=True,
    )

    assert target == target_dir / "hey_orac.onnx"
    assert called["source"] == model_file
    assert called["target"] == target
    assert target.read_text(encoding="utf-8") == "embedded"
    assert not (target_dir / "hey_orac.onnx.data").exists()


def test_embed_onnx_external_data_file_uses_lazy_onnx_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding should be delegated to a lazily imported ONNX module."""
    source = tmp_path / "source.onnx"
    source.write_text("model", encoding="utf-8")
    target = tmp_path / "target.onnx"

    class FakeChecker:
        def __init__(self) -> None:
            self.checked: Path | None = None

        def check_model(self, model_path: str) -> None:
            self.checked = Path(model_path)

    class FakeExternalDataHelper:
        def convert_model_from_external_data(self, model: dict[str, object]) -> None:
            model["embedded"] = True

    class FakeOnnxModule:
        def __init__(self) -> None:
            self.loaded: list[tuple[Path, bool]] = []
            self.saved: list[tuple[dict[str, object], Path]] = []
            self.checker = FakeChecker()
            self.external_data_helper = FakeExternalDataHelper()

        def load(self, path: str, load_external_data: bool = False) -> dict[str, object]:
            self.loaded.append((Path(path), load_external_data))
            return {"path": path, "embedded": False}

        def save_model(self, model: dict[str, object], path: str) -> None:
            self.saved.append((model, Path(path)))
            Path(path).write_text("embedded", encoding="utf-8")

    fake_onnx = FakeOnnxModule()
    monkeypatch.setattr(
        orac_export,
        "_import_onnx_module",
        lambda: fake_onnx,
    )

    orac_export.embed_onnx_external_data_file(source, target)

    assert fake_onnx.loaded == [(source, True), (target, False)]
    assert fake_onnx.saved[0][1] == target
    assert fake_onnx.checker.checked == target
    assert target.read_text(encoding="utf-8") == "embedded"


def test_export_refuses_onnx_when_external_data_is_missing(
    tmp_path: Path,
) -> None:
    """ONNX files that reference external data must export as a pair."""
    project = _project(tmp_path)
    model_file = tmp_path / "hey_orac.onnx"
    model_file.write_text("model", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        export_model_to_orac(project, model_file)


def test_find_generated_models_mirrors_onnx_external_data(
    tmp_path: Path,
) -> None:
    """Mirroring generated models should include ONNX sidecars."""
    project = _project(tmp_path)
    project.openwakeword_output_dir.mkdir(parents=True)
    model_file = project.openwakeword_output_dir / "hey_orac.onnx"
    sidecar_file = project.openwakeword_output_dir / "hey_orac.onnx.data"
    model_file.write_text("model", encoding="utf-8")
    sidecar_file.write_text("weights", encoding="utf-8")

    models = find_generated_models(project)

    assert models == [project.models_dir / "hey_orac.onnx"]
    assert (project.models_dir / "hey_orac.onnx.data").read_text(
        encoding="utf-8"
    ) == "weights"


def test_export_tab_syncs_embed_checkbox_to_model_type() -> None:
    """The embed checkbox should only stay enabled for ONNX models."""
    fake_tab = _FakeExportTab()
    fake_tab.embed_onnx_external_data_var = _FakeBoolVar(True)
    fake_tab.embed_onnx_checkbox = _FakeButton()

    fake_tab.model_path_var.set("/tmp/model.tflite")
    ExportTab._sync_export_options(fake_tab)

    assert fake_tab.embed_onnx_checkbox.state == "disabled"
    assert fake_tab.embed_onnx_external_data_var.get() is False

    fake_tab.model_path_var.set("/tmp/model.onnx")
    ExportTab._sync_export_options(fake_tab)

    assert fake_tab.embed_onnx_checkbox.state == "normal"


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


def test_phrase_part_settings_round_trip(tmp_path: Path) -> None:
    """Positive sample phrase-part settings should persist in project JSON."""
    project = _project(tmp_path)
    project.training_pronunciation_phrase = "Hay O-rack"
    project.training_phrase_parts = ["Hey", "Orac"]
    project.inter_part_silence_min_ms = 90
    project.inter_part_silence_max_ms = 210
    create_project_workspace(project)

    loaded = load_project(project.workspace_dir / "project.json")

    assert loaded.training_pronunciation_phrase == "Hay O-rack"
    assert loaded.training_phrase_parts == ["Hey", "Orac"]
    assert loaded.inter_part_silence_min_ms == 90
    assert loaded.inter_part_silence_max_ms == 210


def test_real_positive_settings_round_trip(tmp_path: Path) -> None:
    """Real positive settings should persist in project JSON."""
    project = _project(tmp_path)
    project.enable_real_positives = True
    project.real_positive_clips_dir = project.workspace_dir / "real_positives"
    project.real_positive_min_count = 12
    project.real_positive_target_percent = 50
    create_project_workspace(project)

    loaded = load_project(project.workspace_dir / "project.json")

    assert loaded.enable_real_positives is True
    assert loaded.real_positive_clips_dir == (
        project.workspace_dir / "real_positives"
    )
    assert loaded.real_positive_min_count == 12
    assert loaded.real_positive_target_percent == 50


def test_real_positive_directory_is_dedicated_project_directory(
    tmp_path: Path,
) -> None:
    """Real positives should always use the dedicated project directory."""
    project = WakeWordProject(
        wake_phrase="Hey Orac",
        model_name="hey_orac",
        workspace_dir=tmp_path / "hey_orac",
        real_positive_clips_dir=tmp_path / "somewhere_else",
    )

    assert project.real_positive_clips_dir == (
        project.workspace_dir / "real_positives"
    )


def test_import_real_positive_wavs_normalises_into_project(
    tmp_path: Path,
) -> None:
    """Imported real positives should become project-local 16 kHz mono WAVs."""
    project = _project(tmp_path)
    create_project_workspace(project)
    source = tmp_path / "source_real.wav"
    tone = np.full((22050, 2), 1200, dtype=np.int16)
    wavfile.write(source, 22050, tone)

    imported = real_positives.import_real_positive_wavs(project, [source])

    assert len(imported) == 1
    assert imported[0].parent == project.real_positive_clips_dir
    sample_rate, data = wavfile.read(imported[0])
    assert sample_rate == 16000
    assert data.ndim == 1
    assert data.dtype == np.int16


def test_import_multiple_real_positive_wavs(
    tmp_path: Path,
) -> None:
    """Import should accept several real positive WAV files at once."""
    project = _project(tmp_path)
    create_project_workspace(project)
    sources = []
    for index in range(3):
        source = tmp_path / f"user_real_{index}.wav"
        wavfile.write(
            source,
            16000,
            np.full(16000, 1000 + index, dtype=np.int16),
        )
        sources.append(source)

    imported = real_positives.import_real_positive_wavs(project, sources)

    assert len(imported) == 3
    assert all(path.parent == project.real_positive_clips_dir for path in imported)
    assert sorted(path.name for path in imported) == [
        "user_real_0.wav",
        "user_real_1.wav",
        "user_real_2.wav",
    ]


def test_validate_real_positive_wav_reports_bad_audio(
    tmp_path: Path,
) -> None:
    """Real positive validation should reject empty WAV clips."""
    wav_path = tmp_path / "empty.wav"
    wavfile.write(wav_path, 16000, np.array([], dtype=np.int16))

    result = real_positives.validate_real_positive_clip(wav_path)

    assert result.ok is False
    assert any("empty" in message for message in result.messages)


def test_import_near_miss_wavs_normalises_into_project(
    tmp_path: Path,
) -> None:
    """Near-miss imports should become project-local 16 kHz mono WAVs."""
    project = _project(tmp_path)
    create_project_workspace(project)
    source = tmp_path / "source_near_miss.wav"
    tone = np.full((22050, 2), 1200, dtype=np.int16)
    wavfile.write(source, 22050, tone)

    imported = near_misses.import_near_miss_wavs(project, [source])

    assert len(imported) == 1
    assert imported[0].parent == project.near_miss_clips_dir
    sample_rate, data = wavfile.read(imported[0])
    assert sample_rate == 16000
    assert data.ndim == 1
    assert data.dtype == np.int16


def test_generate_synthetic_near_miss_wavs_uses_negative_phrases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic near-miss generation should be driven by negative phrases."""
    project = _project(tmp_path)
    project.custom_negative_phrases = ["Hey Oracle", "Oracle"]
    project.piper_sample_generator_path.mkdir(parents=True)
    (project.piper_sample_generator_path / "generate_samples.py").write_text(
        "",
        encoding="utf-8",
    )
    project.piper_voice_model_path.parent.mkdir(parents=True, exist_ok=True)
    project.piper_voice_model_path.write_text("model", encoding="utf-8")
    create_project_workspace(project)

    def fake_synthesise(phrase: str, _project: WakeWordProject, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(
            output_dir / "0.wav",
            16000,
            np.full(16000, 1000 if phrase == "Hey Oracle" else 800, dtype=np.int16),
        )

    monkeypatch.setattr(near_misses, "_synthesise_phrase", fake_synthesise)

    generated = near_misses.generate_synthetic_near_miss_wavs(project)

    assert len(generated) == 2
    assert all(path.parent == project.near_miss_clips_dir for path in generated)
    assert generated[0].name.startswith("synthetic_near_miss_0000_")
    assert generated[1].name.startswith("synthetic_near_miss_0001_")


def test_near_miss_score_output_summarises_results() -> None:
    """Directory scoring should report summary stats and activated clips."""
    results = [
        model_tester.WavTestResult(
            model_path=Path("model.onnx"),
            wav_path=Path("/tmp/near_miss_1.wav"),
            model_name="hey_orac",
            threshold=0.75,
            max_score=0.31,
            frame_count=4,
        ),
        model_tester.WavTestResult(
            model_path=Path("model.onnx"),
            wav_path=Path("/tmp/near_miss_2.wav"),
            model_name="hey_orac",
            threshold=0.75,
            max_score=0.82,
            frame_count=4,
        ),
    ]

    rendered = model_tester.render_directory_results(
        "Similar phrases that should not wake Orac",
        Path("/tmp/near_misses"),
        results,
    )

    assert "Files evaluated: 2" in rendered
    assert "Activated: 1/2" in rendered
    assert "Min score: 0.3100" in rendered
    assert "Average score:" in rendered
    assert "Max score: 0.8200" in rendered
    assert "Activated clips:" in rendered
    assert "near_miss_2.wav" in rendered


def test_near_miss_scoring_empty_folder_shows_guidance(
    tmp_path: Path,
) -> None:
    """Score Near Misses should explain what belongs in the folder."""
    project = _project(tmp_path)
    create_project_workspace(project)
    tab = test_tab.TestTab.__new__(test_tab.TestTab)
    tab.app = _FakeApp()
    tab.app.project = project
    tab.model_path_var = _FakeMutablePathVar(
        str(project.openwakeword_output_dir / f"{project.model_name}.onnx")
    )
    tab.wav_path_var = _FakeMutablePathVar("")
    tab.threshold_var = _FakeMutablePathVar("0.75")
    tab.status_var = _FakeStatusVar()
    tab.output = _FakeChecksOutput("")
    tab.output_text = ""

    test_tab.TestTab.score_near_misses(tab)

    assert "No near-miss WAV files were found" in tab.output_text


def test_stage_real_positives_copies_into_positive_train(
    tmp_path: Path,
) -> None:
    """Real positives should be staged as extra positive training clips."""
    project = _project(tmp_path)
    create_project_workspace(project)
    real_clip = project.real_positive_clips_dir / "user_001.wav"
    stereo = np.full((44100, 2), 1000, dtype=np.int16)
    wavfile.write(real_clip, 44100, stereo)
    synthetic_clip_dir = (
        project.openwakeword_output_dir / project.model_name / "positive_train"
    )
    synthetic_clip_dir.mkdir(parents=True)
    for index in range(9):
        (synthetic_clip_dir / f"{index}.wav").write_bytes(b"synthetic")
    project.real_positive_target_percent = 10

    staged = real_positives.stage_real_positives_for_training(project)

    assert staged == 1
    assert real_clip.exists()
    assert (synthetic_clip_dir / "0.wav").exists()
    staged_clip = synthetic_clip_dir / "real_positive_0000_00.wav"
    assert staged_clip.exists()
    sample_rate, data = wavfile.read(staged_clip)
    assert sample_rate == 16000
    assert data.ndim == 1
    assert np.max(np.abs(data)) < 2000


def test_stage_real_positives_does_not_clip_stereo_int16(
    tmp_path: Path,
) -> None:
    """Stereo int16 real positives should be scaled before mono mixing."""
    project = _project(tmp_path)
    create_project_workspace(project)
    real_clip = project.real_positive_clips_dir / "user_stereo.wav"
    left = np.full(44100, 1200, dtype=np.int16)
    right = np.full(44100, -800, dtype=np.int16)
    wavfile.write(real_clip, 44100, np.column_stack([left, right]))
    (project.openwakeword_output_dir / project.model_name / "positive_train").mkdir(
        parents=True,
    )
    project.real_positive_target_percent = 10

    real_positives.stage_real_positives_for_training(project)

    staged_clip = (
        project.openwakeword_output_dir
        / project.model_name
        / "positive_train"
        / "real_positive_0000_00.wav"
    )
    sample_rate, data = wavfile.read(staged_clip)
    assert sample_rate == 16000
    assert data.ndim == 1
    assert np.max(np.abs(data)) < 1200
    assert np.max(np.abs(data)) > 100


def test_stage_real_positives_uses_target_training_mix(
    tmp_path: Path,
) -> None:
    """Real positive staging should approximate the requested mix."""
    project = _project(tmp_path)
    project.real_positive_target_percent = 50
    create_project_workspace(project)
    real_clip = project.real_positive_clips_dir / "user_001.wav"
    wavfile.write(real_clip, 16000, np.full(16000, 1000, dtype=np.int16))
    synthetic_clip_dir = (
        project.openwakeword_output_dir / project.model_name / "positive_train"
    )
    synthetic_clip_dir.mkdir(parents=True)
    for index in range(20):
        (synthetic_clip_dir / f"{index}.wav").write_bytes(b"synthetic")

    staged = real_positives.stage_real_positives_for_training(project)

    assert staged == 20
    assert len(list(synthetic_clip_dir.glob("real_positive_*.wav"))) == 20


def test_stage_real_positives_can_exclude_synthetic_train_clips(
    tmp_path: Path,
) -> None:
    """A 100 percent real mix should exclude synthetic positive training clips."""
    project = _project(tmp_path)
    project.real_positive_target_percent = 100
    create_project_workspace(project)
    real_clip = project.real_positive_clips_dir / "user_001.wav"
    wavfile.write(real_clip, 16000, np.full(16000, 1000, dtype=np.int16))
    synthetic_clip_dir = (
        project.openwakeword_output_dir / project.model_name / "positive_train"
    )
    synthetic_clip_dir.mkdir(parents=True)
    for index in range(20):
        (synthetic_clip_dir / f"{index}.wav").write_bytes(b"synthetic")

    staged = real_positives.stage_real_positives_for_training(project)

    assert staged == 20
    assert not list(synthetic_clip_dir.glob("[0-9]*.wav"))
    assert len(list(synthetic_clip_dir.glob("real_positive_*.wav"))) == 20


def test_saved_project_can_be_discovered_after_restart(
    tmp_path: Path,
) -> None:
    """Saved projects should be discovered from the projects root."""
    project = _project(tmp_path)
    create_project_workspace(project)

    discovered = discover_projects(tmp_path)

    assert [item.model_name for item in discovered] == ["hey_orac"]
    assert discovered[0].workspace_dir == project.workspace_dir


def test_delete_project_workspace_removes_only_valid_project(
    tmp_path: Path,
) -> None:
    """Project deletion should remove a saved project workspace."""
    project = _project(tmp_path)
    create_project_workspace(project)
    (project.logs_dir / "train_model.log").write_text(
        "log",
        encoding="utf-8",
    )

    deleted = delete_project_workspace(project)

    assert deleted is True
    assert not project.workspace_dir.exists()


def test_delete_project_workspace_rejects_non_project_directory(
    tmp_path: Path,
) -> None:
    """Project deletion should not remove arbitrary directories."""
    project = _project(tmp_path)
    project.workspace_dir.mkdir(parents=True)
    (project.workspace_dir / "notes.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="project.json"):
        delete_project_workspace(project)

    assert project.workspace_dir.exists()


def test_open_project_uses_existing_load_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening a project from the UI path should call load_project."""
    expected_project = _project(tmp_path)
    expected_path = expected_project.workspace_dir / "project.json"
    calls: list[Path] = []
    observed_project_before_apply: WakeWordProject | None = None

    def fake_load_project(project_path: Path) -> WakeWordProject:
        calls.append(project_path)
        return expected_project

    fake_tab = _FakeProjectTab()
    fake_tab.selected_project_var = _FakeMutablePathVar("")
    monkeypatch.setattr(project_tab, "load_project", fake_load_project)

    def fake_apply_project(project: WakeWordProject) -> None:
        nonlocal observed_project_before_apply
        observed_project_before_apply = fake_tab.app.project
        fake_tab.applied_project = project

    fake_tab.apply_project = fake_apply_project  # type: ignore[method-assign]

    loaded = ProjectTab.open_project(fake_tab, expected_path)

    assert loaded is expected_project
    assert calls == [expected_path]
    assert observed_project_before_apply is expected_project
    assert fake_tab.applied_project is expected_project
    assert fake_tab.app.project is expected_project
    assert fake_tab.selected_project_var.get() == "hey_orac (Hey Orac)"


def test_open_selected_project_loads_immediately_from_dropdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a project from the dropdown should open it immediately."""
    project = _project(tmp_path)
    project_path = project.workspace_dir / "project.json"
    calls: list[Path] = []

    def fake_load_project(project_path_arg: Path) -> WakeWordProject:
        calls.append(project_path_arg)
        return project

    fake_tab = _FakeProjectTab()
    fake_tab.selected_project_var = _FakeMutablePathVar("")
    fake_tab.available_project_paths = {"hey_orac (Hey Orac)": project_path}
    fake_tab.status_var = _FakeStatusVar()
    fake_tab.open_project = lambda project_path_arg: ProjectTab.open_project(
        fake_tab,
        project_path_arg,
    )
    monkeypatch.setattr(project_tab, "load_project", fake_load_project)

    ProjectTab.open_selected_project(fake_tab, "hey_orac (Hey Orac)")

    assert calls == [project_path]
    assert fake_tab.app.project is project
    assert fake_tab.applied_project is project
    assert fake_tab.selected_project_var.get() == "hey_orac (Hey Orac)"
    assert fake_tab.status_var.value == f"Project loaded: {project.workspace_dir}."


def test_refresh_projects_keeps_current_project_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refreshing projects should keep the active project selected."""
    project = _project(tmp_path)
    project.workspace_dir = tmp_path / "projects" / "hey_orac"
    other_project = _project(tmp_path)
    other_project.model_name = "computer"
    other_project.wake_phrase = "Computer"
    other_project.workspace_dir = tmp_path / "projects" / "computer"
    fake_tab = _FakeProjectTab()
    fake_tab.app.project = project
    fake_tab.project_menu = _FakeProjectMenu()
    fake_tab.selected_project_var = _FakeMutablePathVar("computer (Computer)")

    monkeypatch.setattr(
        project_tab,
        "discover_projects",
        lambda *_args, **_kwargs: [other_project, project],
    )

    ProjectTab.refresh_projects(fake_tab)

    assert fake_tab.project_menu.values == [
        "computer (Computer)",
        "hey_orac (Hey Orac)",
    ]
    assert fake_tab.selected_project_var.get() == "hey_orac (Hey Orac)"


def test_open_project_folder_uses_workspace_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Open Folder action should open the selected project workspace."""
    project = _project(tmp_path)
    create_project_workspace(project)
    fake_tab = _FakeProjectTab()
    fake_tab.app.project = project
    fake_tab.selected_project_var = _FakeMutablePathVar(
        "hey_orac (Hey Orac)"
    )
    fake_tab.available_project_paths = {
        "hey_orac (Hey Orac)": project.workspace_dir / "project.json",
    }
    fake_tab.status_var = _FakeStatusVar()
    opened: list[Path] = []

    monkeypatch.setattr(project_tab, "_open_folder", opened.append)

    ProjectTab.open_selected_project_folder(fake_tab)

    assert opened == [project.workspace_dir]
    assert fake_tab.status_var.value == (
        f"Opened project folder: {project.workspace_dir}."
    )


def test_loaded_project_fields_are_mapped_to_project_form(
    tmp_path: Path,
) -> None:
    """Loaded project data should populate form values."""
    project = _project(tmp_path)
    project.wake_phrase = "Computer"
    project.model_name = "computer"
    project.openwakeword_repo = tmp_path / "custom_openwakeword"
    project.piper_voice_model_path = tmp_path / "voices" / "en_US.onnx"
    project.background_paths = [tmp_path / "bg_one", tmp_path / "bg_two"]
    project.custom_negative_phrases = ["compute", "commuter"]
    project.profile = "balanced"

    values = project_form_values(project)

    assert values["wake_phrase"] == "Computer"
    assert values["model_name"] == "computer"
    assert values["workspace_root"] == str(project.workspace_dir.parent)
    assert values["openwakeword_repo"] == str(
        tmp_path / "custom_openwakeword"
    )
    assert values["piper_voice_model_path"] == str(
        tmp_path / "voices" / "en_US.onnx"
    )
    assert values["training_pronunciation_phrase"] == ""
    assert values["training_phrase_parts"] == ""
    assert values["inter_part_silence_min_ms"] == "80"
    assert values["inter_part_silence_max_ms"] == "250"
    assert values["enable_real_positives"] == "true"
    assert values["real_positive_clips_dir"] == str(
        project.workspace_dir / "real_positives"
    )
    assert values["real_positive_min_count"] == "20"
    assert values["real_positive_target_percent"] == "30%"
    assert values["background_paths"] == (
        f"{tmp_path / 'bg_one'},{tmp_path / 'bg_two'}"
    )
    assert values["custom_negative_phrases"] == "compute,commuter"
    assert values["profile"] == "balanced"


def test_new_project_resets_form_and_clears_active_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New Project should clear project-specific fields without saving."""
    defaults = {
        "wake_phrase": "",
        "model_name": "",
        "workspace_root": str(tmp_path / "projects"),
        "openwakeword_repo": str(tmp_path / "external" / "openWakeWord"),
        "orac_repo": str(tmp_path / "external" / "Orac"),
        "piper_sample_generator_path": str(tmp_path / "piper"),
        "piper_voice_model_path": "",
        "training_pronunciation_phrase": "",
        "training_phrase_parts": "",
        "inter_part_silence_min_ms": "80",
        "inter_part_silence_max_ms": "250",
        "enable_real_positives": "true",
        "real_positive_clips_dir": "",
        "real_positive_min_count": "20",
        "real_positive_target_percent": "30%",
        "background_paths": str(tmp_path / "background"),
        "rir_paths": str(tmp_path / "rir"),
        "negative_feature_data_files": "",
        "false_positive_validation_data_path": "",
        "custom_negative_phrases": "",
        "profile": "quick",
    }
    tab = ProjectTab.__new__(ProjectTab)
    tab.app = _FakeApp()
    tab.app.project = _project(tmp_path)
    tab.wake_phrase_var = _FakeMutablePathVar("Computer")
    tab.model_name_var = _FakeMutablePathVar("computer")
    tab.workspace_root_var = _FakeMutablePathVar(str(tmp_path / "old"))
    tab.oww_path_var = _FakeMutablePathVar(str(tmp_path / "old_oww"))
    tab.orac_path_var = _FakeMutablePathVar(str(tmp_path / "old_orac"))
    tab.piper_path_var = _FakeMutablePathVar(str(tmp_path / "old_piper"))
    tab.piper_model_path_var = _FakeMutablePathVar(
        str(tmp_path / "voices" / "old.onnx")
    )
    tab.training_pronunciation_phrase_var = _FakeMutablePathVar("old phrase")
    tab.training_phrase_parts_var = _FakeMutablePathVar("old | parts")
    tab.inter_part_silence_min_var = _FakeMutablePathVar("1")
    tab.inter_part_silence_max_var = _FakeMutablePathVar("2")
    tab.enable_real_positives_var = _FakeBoolVar(False)
    tab.real_positive_dir_var = _FakeMutablePathVar(str(tmp_path / "old_real"))
    tab.real_positive_min_count_var = _FakeMutablePathVar("5")
    tab.real_positive_target_percent_var = _FakeMutablePathVar("80%")
    tab.background_paths_var = _FakeMutablePathVar(str(tmp_path / "old_bg"))
    tab.rir_paths_var = _FakeMutablePathVar(str(tmp_path / "old_rir"))
    tab.negative_feature_paths_var = _FakeMutablePathVar(
        str(tmp_path / "old.npy")
    )
    tab.validation_path_var = _FakeMutablePathVar(str(tmp_path / "old_fp.npy"))
    tab.negatives_var = _FakeMutablePathVar("computer")
    tab.profile_var = _FakeMutablePathVar("balanced")
    tab.status_var = _FakeStatusVar()
    refresh_saw_cleared_project = False

    def fake_refresh_feature_bundle(**_kwargs: object) -> None:
        nonlocal refresh_saw_cleared_project
        refresh_saw_cleared_project = tab.app.project is None

    tab.refresh_feature_bundle = fake_refresh_feature_bundle  # type: ignore[method-assign]
    monkeypatch.setattr(
        project_tab,
        "_new_project_form_values",
        lambda: defaults,
    )

    ProjectTab.new_project(tab)

    assert tab.app.project is None
    assert refresh_saw_cleared_project is True
    assert tab.wake_phrase_var.get() == defaults["wake_phrase"]
    assert tab.model_name_var.get() == defaults["model_name"]
    assert tab.workspace_root_var.get() == defaults["workspace_root"]
    assert tab.oww_path_var.get() == defaults["openwakeword_repo"]
    assert tab.orac_path_var.get() == defaults["orac_repo"]
    assert (
        tab.piper_path_var.get()
        == defaults["piper_sample_generator_path"]
    )
    assert tab.piper_model_path_var.get() == ""
    assert tab.training_pronunciation_phrase_var.get() == ""
    assert tab.training_phrase_parts_var.get() == ""
    assert tab.inter_part_silence_min_var.get() == "80"
    assert tab.inter_part_silence_max_var.get() == "250"
    assert tab.enable_real_positives_var.get() is True
    assert tab.real_positive_dir_var.get() == ""
    assert tab.real_positive_min_count_var.get() == "20"
    assert tab.real_positive_target_percent_var.get() == "30%"
    assert tab.background_paths_var.get() == defaults["background_paths"]
    assert tab.rir_paths_var.get() == defaults["rir_paths"]
    assert tab.negative_feature_paths_var.get() == ""
    assert tab.validation_path_var.get() == ""
    assert (
        tab.negatives_var.get()
        == defaults["custom_negative_phrases"]
    )
    assert tab.profile_var.get() == defaults["profile"]
    assert tab.status_var.value == "New project form ready."


def test_delete_project_clears_active_project_and_refreshes_dropdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the current project should refresh saved project choices."""
    project = _project(tmp_path)
    other_project = _project(tmp_path)
    other_project.model_name = "computer"
    other_project.wake_phrase = "Computer"
    other_project.workspace_dir = tmp_path / "computer"
    create_project_workspace(project)
    create_project_workspace(other_project)
    tab = ProjectTab.__new__(ProjectTab)
    tab.app = _FakeApp()
    tab.app.project = project
    tab.status_var = _FakeStatusVar()
    tab.project_menu = _FakeProjectMenu()
    tab.selected_project_var = _FakeMutablePathVar("")
    tab.available_project_paths = {}
    form_was_reset = False
    feature_bundle_refreshed = False

    def fake_apply_form_values(_values: dict[str, str]) -> None:
        nonlocal form_was_reset
        form_was_reset = True

    def fake_refresh_feature_bundle(**_kwargs: object) -> None:
        nonlocal feature_bundle_refreshed
        feature_bundle_refreshed = True

    monkeypatch.setattr(
        project_tab.messagebox,
        "askyesno",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        project_tab,
        "discover_projects",
        lambda: discover_projects(tmp_path),
    )
    tab._apply_form_values = fake_apply_form_values  # type: ignore[method-assign]
    tab.refresh_feature_bundle = fake_refresh_feature_bundle  # type: ignore[method-assign]

    ProjectTab.delete_project(tab)

    assert tab.app.project is None
    assert not project.workspace_dir.exists()
    assert other_project.workspace_dir.exists()
    assert form_was_reset is True
    assert feature_bundle_refreshed is True
    assert tab.project_menu.values == ["computer (Computer)"]
    assert tab.selected_project_var.get() == "computer (Computer)"
    assert "hey_orac (Hey Orac)" not in tab.project_menu.values
    assert tab.status_var.value == f"Project deleted: {project.workspace_dir}."


def test_piper_model_selection_updates_loaded_project_but_not_disk(
    tmp_path: Path,
) -> None:
    """Selecting a Piper model should only update the active project in memory."""
    project = _project(tmp_path)
    project.piper_voice_model_path = Path("")
    create_project_workspace(project)
    model_path = tmp_path / "voices" / "en_US.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("model", encoding="utf-8")
    fake_tab = _FakePiperModelTab(project, model_path)

    ProjectTab._sync_current_project_piper_model_path(fake_tab)

    reloaded = load_project(project.workspace_dir / "project.json")

    assert project.piper_voice_model_path == model_path
    assert reloaded.piper_voice_model_path == Path("")


def test_new_project_paths_use_managed_defaults() -> None:
    """New project paths should still default under the managed root."""
    assert project_dir_for_model("new_project") == (
        Path("~/WakeLab").expanduser().resolve()
        / "projects"
        / "new_project"
    )


def test_theme_path_points_to_bundled_theme() -> None:
    """The app should load the bundled CustomTkinter theme."""
    assert THEME_PATH.parent.name == "themes"
    assert THEME_PATH.exists()


def test_set_appearance_mode_forwards_to_customtkinter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing appearance mode should call the CustomTkinter setter."""
    modes: list[str] = []

    def fake_set_appearance_mode(mode: str) -> None:
        modes.append(mode)

    monkeypatch.setattr(
        main_window.ctk,
        "set_appearance_mode",
        fake_set_appearance_mode,
    )

    OracWakeLabApp.set_appearance_mode(object(), "Light")

    assert modes == ["Light"]
    assert DEFAULT_APPEARANCE_MODE == "Dark"


def test_dependency_check_results_are_structured(tmp_path: Path) -> None:
    """Dependency checks should return structured statuses."""
    project = _project(tmp_path)
    # Mock piper generator layout
    (project.piper_sample_generator_path / "piper_sample_generator").mkdir(parents=True)
    (project.piper_sample_generator_path / "piper_sample_generator" / "__main__.py").write_text("")
    (project.piper_sample_generator_path / "piper_sample_generator" / "augment.py").write_text("")
    
    # Mock openwakeword layout
    project.openwakeword_repo.mkdir(parents=True)
    (project.openwakeword_repo / "openwakeword").mkdir()
    (project.openwakeword_repo / "openwakeword" / "train.py").write_text("")

    results = run_dependency_checks(project)

    assert results
    assert all(isinstance(result, ValidationResult) for result in results)
    assert {result.status for result in results}.issubset(
        {"pass", "warn", "fail"}
    )
    assert any(result.name == "import torchinfo" for result in results)
    assert any(
        result.name == "openWakeWord training assets" for result in results
    )
    assert any(result.name == "import torchmetrics" for result in results)
    assert any(result.name == "import pronouncing" for result in results)
    assert any(result.name == "import dp" for result in results)
    assert any(result.name == "import torch_audiomentations" for result in results)
    assert any(result.name == "import speechbrain" for result in results)
    assert any(result.name == "import mutagen" for result in results)
    assert any(result.name == "import acoustics" for result in results)
    assert any(result.name == "import onnxscript" for result in results)


def test_checks_tab_copies_output_to_clipboard() -> None:
    """Copying the checks output should place the rendered text on clipboard."""
    fake_tab = _FakeChecksTab()

    ChecksTab.copy_output_to_clipboard(fake_tab)

    assert fake_tab.clipboard_cleared is True
    assert fake_tab.clipboard_text == fake_tab.output_text
    assert fake_tab.update_called is True


def test_training_tab_copies_log_to_clipboard() -> None:
    """Copying the training log should place the rendered text on clipboard."""
    fake_tab = _FakeClipboardTab("train log")

    training_tab.TrainingTab.copy_log_to_clipboard(fake_tab)

    assert fake_tab.clipboard_cleared is True
    assert fake_tab.clipboard_text == "train log"
    assert fake_tab.update_called is True


def test_training_tab_clears_log_when_no_job_is_running() -> None:
    """The Train tab clear action should empty the visible log when idle."""
    fake_tab = _FakeClipboardTab("train log")
    fake_tab.runner = _FakeRunner(is_running=False)

    training_tab.TrainingTab.clear_log(fake_tab)

    assert fake_tab.log.text == ""
    assert fake_tab.log.state == "disabled"


def test_training_tab_does_not_clear_log_while_job_is_running() -> None:
    """The Train tab clear action should do nothing during a running job."""
    fake_tab = _FakeClipboardTab("train log")
    fake_tab.runner = _FakeRunner(is_running=True)

    training_tab.TrainingTab.clear_log(fake_tab)

    assert fake_tab.log.text == "train log"


def test_test_tab_copies_output_to_clipboard() -> None:
    """Copying the test output should place the rendered text on clipboard."""
    fake_tab = _FakeTestTab()

    test_tab.TestTab.copy_output_to_clipboard(fake_tab)

    assert fake_tab.clipboard_cleared is True
    assert fake_tab.clipboard_text == fake_tab.output_text
    assert fake_tab.update_called is True


def test_model_tester_summarizes_predictions() -> None:
    """Prediction summaries should report the highest frame score."""
    model_name, score, frames = model_tester.summarize_predictions(
        [
            {"hey_orac": 0.1},
            {"hey_orac": 0.82},
            {"hey_orac": 0.4},
        ]
    )

    assert model_name == "hey_orac"
    assert score == pytest.approx(0.82)
    assert frames == 3


def test_model_tester_result_render_marks_activation(tmp_path: Path) -> None:
    """Rendered test results should include threshold status."""
    result = model_tester.WavTestResult(
        model_path=tmp_path / "hey_orac.onnx",
        wav_path=tmp_path / "sample.wav",
        model_name="hey_orac",
        threshold=0.75,
        max_score=0.9,
        frame_count=12,
    )

    rendered = result.render()

    assert result.activated is True
    assert "ACTIVATED" in rendered
    assert "Max score: 0.9000" in rendered


def test_test_tab_model_browser_starts_in_project_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model browser should open in the project model output directory."""
    project = _project(tmp_path)
    fake_tab = _FakeTestTab(project)
    captured: dict[str, str] = {}

    def fake_askopenfilename(**kwargs: object) -> str:
        captured["initialdir"] = str(kwargs["initialdir"])
        return ""

    monkeypatch.setattr(
        test_tab.filedialog,
        "askopenfilename",
        fake_askopenfilename,
    )

    test_tab.TestTab.browse_model(fake_tab)

    assert captured["initialdir"] == str(project.openwakeword_output_dir)


def test_test_tab_refresh_sets_project_model_and_wav_defaults(
    tmp_path: Path,
) -> None:
    """Refresh should populate defaults from the active project."""
    project = _project(tmp_path)
    positive_dir = (
        project.openwakeword_output_dir
        / project.model_name
        / "positive_test"
    )
    positive_dir.mkdir(parents=True)
    (positive_dir / "0.wav").write_bytes(b"wav")
    fake_tab = _FakeTestTab(project)

    test_tab.TestTab.refresh_from_project(fake_tab)

    assert fake_tab.model_path_var.get() == str(
        project.openwakeword_output_dir / f"{project.model_name}.onnx"
    )
    assert fake_tab.wav_path_var.get() == str(positive_dir / "0.wav")


def test_test_tab_refresh_replaces_stale_other_project_wav(
    tmp_path: Path,
) -> None:
    """Refresh should replace stale WAV paths from another project."""
    project = _project(tmp_path)
    create_project_workspace(project)
    current_positive_dir = (
        project.openwakeword_output_dir
        / project.model_name
        / "positive_test"
    )
    current_positive_dir.mkdir(parents=True)
    (current_positive_dir / "0.wav").write_bytes(b"wav")
    other_project = _project(tmp_path)
    other_project.model_name = "other"
    other_project.workspace_dir = tmp_path / "other"
    create_project_workspace(other_project)
    stale_dir = (
        other_project.openwakeword_output_dir
        / other_project.model_name
        / "positive_test"
    )
    stale_dir.mkdir(parents=True)
    fake_tab = _FakeTestTab(project)
    fake_tab.wav_path_var.set(str(stale_dir / "sample.wav"))

    test_tab.TestTab.refresh_from_project(fake_tab)

    assert fake_tab.wav_path_var.get() == str(current_positive_dir / "0.wav")


def test_test_tab_wav_browser_starts_in_real_positives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAV browser should open in project real positives by default."""
    project = _project(tmp_path)
    fake_tab = _FakeTestTab(project)
    captured: dict[str, str] = {}

    def fake_askopenfilename(**kwargs: object) -> str:
        captured["initialdir"] = str(kwargs["initialdir"])
        return ""

    monkeypatch.setattr(
        test_tab.filedialog,
        "askopenfilename",
        fake_askopenfilename,
    )

    test_tab.TestTab.browse_wav(fake_tab)

    assert captured["initialdir"] == str(project.real_positive_clips_dir)


def test_test_tab_wav_browser_keeps_existing_wav_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAV browser should preserve a current valid WAV directory."""
    project = _project(tmp_path)
    existing_dir = tmp_path / "existing_wavs"
    existing_dir.mkdir()
    fake_tab = _FakeTestTab(project)
    fake_tab.wav_path_var.set(str(existing_dir / "sample.wav"))
    captured: dict[str, str] = {}

    def fake_askopenfilename(**kwargs: object) -> str:
        captured["initialdir"] = str(kwargs["initialdir"])
        return ""

    monkeypatch.setattr(
        test_tab.filedialog,
        "askopenfilename",
        fake_askopenfilename,
    )

    test_tab.TestTab.browse_wav(fake_tab)

    assert captured["initialdir"] == str(existing_dir)


def test_test_tab_wav_browser_resets_other_project_wav_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAV browser should not keep a WAV path from another project."""
    project = _project(tmp_path)
    create_project_workspace(project)
    other_project = _project(tmp_path)
    other_project.model_name = "other"
    other_project.workspace_dir = tmp_path / "other"
    create_project_workspace(other_project)
    stale_dir = (
        other_project.openwakeword_output_dir
        / other_project.model_name
        / "positive_test"
    )
    stale_dir.mkdir(parents=True)
    current_positive_dir = (
        project.openwakeword_output_dir
        / project.model_name
        / "positive_test"
    )
    current_positive_dir.mkdir(parents=True)
    fake_tab = _FakeTestTab(project)
    fake_tab.wav_path_var.set(str(stale_dir / "sample.wav"))
    captured: dict[str, str] = {}

    def fake_askopenfilename(**kwargs: object) -> str:
        captured["initialdir"] = str(kwargs["initialdir"])
        return ""

    monkeypatch.setattr(
        test_tab.filedialog,
        "askopenfilename",
        fake_askopenfilename,
    )

    test_tab.TestTab.browse_wav(fake_tab)

    assert captured["initialdir"] == str(current_positive_dir)


def test_export_tab_model_browser_starts_in_project_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export model browser should open under the project model directory."""
    project = _project(tmp_path)
    project.models_dir.mkdir(parents=True)
    fake_tab = _FakeExportTab(project)
    captured: dict[str, str] = {}

    def fake_askopenfilename(**kwargs: object) -> str:
        captured["initialdir"] = str(kwargs["initialdir"])
        return ""

    monkeypatch.setattr(
        export_tab.filedialog,
        "askopenfilename",
        fake_askopenfilename,
    )

    ExportTab.browse_model(fake_tab)

    assert captured["initialdir"] == str(project.models_dir)


def test_main_window_set_project_refreshes_test_tab(tmp_path: Path) -> None:
    """Loading a project should refresh project-derived Test tab fields."""
    project = _project(tmp_path)
    app = object.__new__(OracWakeLabApp)
    app.project_label = _FakeLabel()
    app.test_tab = _FakeRefreshTab()
    app.export_tab = _FakeRefreshTab()

    OracWakeLabApp.set_project(app, project)

    assert app.current_project == project
    assert app.test_tab.refreshed is True
    assert app.export_tab.refreshed is True


def test_main_window_clear_project_refreshes_project_derived_tabs(
    tmp_path: Path,
) -> None:
    """Clearing a project should refresh tabs that cache project paths."""
    app = object.__new__(OracWakeLabApp)
    app.current_project = _project(tmp_path)
    app.project_label = _FakeLabel()
    app.test_tab = _FakeRefreshTab()
    app.export_tab = _FakeRefreshTab()

    OracWakeLabApp.clear_project(app)

    assert app.current_project is None
    assert app.project_label.text == "No project loaded."
    assert app.test_tab.refreshed is True
    assert app.export_tab.refreshed is True


def test_onnx_tf_missing_tensorflow_addons_is_reported_as_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing TensorFlow Addons should warn on Python 3.12."""

    def fake_import(module_name: str) -> object:
        if module_name == "onnx_tf":
            raise ModuleNotFoundError(
                "No module named 'tensorflow_addons'",
                name="tensorflow_addons",
            )
        return object()

    monkeypatch.setattr(dependency_checks.importlib, "import_module", fake_import)
    monkeypatch.setattr(dependency_checks.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(dependency_checks.platform, "python_version", lambda: "3.12.0")

    result = dependency_checks._check_onnx_tf_import()

    assert result.status == "warn"
    assert "tensorflow_addons" in result.message
    assert "convert_tflite" in result.blocks


def test_build_training_job_uses_sys_executable(tmp_path: Path) -> None:
    """Subprocess commands should use the current Python interpreter."""
    project = _project(tmp_path)
    job = build_training_job(project, "generate_clips")

    assert job.command[0] == sys.executable
    assert job.command[1] == "-m"
    assert job.command[2] == "orac_wake_lab.services.openwakeword_train_runner"


def test_openwakeword_runner_patches_store_true_string_defaults() -> None:
    """Upstream string defaults should become real booleans."""
    openwakeword_train_runner._patch_store_true_defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert_to_tflite", action="store_true", default="False")

    args = parser.parse_args([])

    assert args.convert_to_tflite is False


def test_openwakeword_runner_filters_known_warning_noise() -> None:
    """Runner should hide known third-party warning spam."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("default")
        openwakeword_train_runner._filter_noisy_third_party_warnings()
        warnings.warn_explicit(
            "pkg_resources is deprecated as an API",
            UserWarning,
            filename="pronouncing/__init__.py",
            lineno=3,
            module="pronouncing.__init__",
        )
        warnings.warn_explicit(
            "real training problem",
            RuntimeWarning,
            filename="openwakeword/train.py",
            lineno=1,
            module="openwakeword.train",
        )

    assert len(captured) == 1
    assert str(captured[0].message) == "real training problem"


def test_build_training_job_generate_clips_uses_piper_model_and_counts(
    tmp_path: Path,
) -> None:
    """Generate clips should invoke the upstream generate phase."""
    project = _project(tmp_path)
    project.piper_voice_model_path = tmp_path / "voices" / "en_US.onnx"
    job = build_training_job(project, "generate_clips")

    assert job.command[:3] == [
        sys.executable,
        "-m",
        "orac_wake_lab.services.openwakeword_train_runner",
    ]
    assert "--generate_clips" in job.command
    assert "--training_config" in job.command
    assert job.command[4].endswith("generate_clips.json")


def test_blank_phrase_parts_preserve_whole_phrase_generation(
    tmp_path: Path,
) -> None:
    """Without pronunciation controls, existing Piper generation is unchanged."""
    project = _project(tmp_path)
    project.piper_sample_generator_path.mkdir(parents=True)
    (project.piper_sample_generator_path / "generate_samples.py").write_text(
        "",
        encoding="utf-8",
    )

    job = build_training_job(project, "generate_clips")
    runtime_config = json.loads(
        (project.config_dir / "_runtime" / "generate_clips.json").read_text(
            encoding="utf-8"
        )
    )

    assert runtime_config["piper_sample_generator_path"] == str(
        project.piper_sample_generator_path
    )
    assert job.env["WAKELAB_TRAINING_PHRASE_PARTS_JSON"] == "[]"
    assert job.env["WAKELAB_TRAINING_PRONUNCIATION_PHRASE"] == ""
    assert runtime_config["target_phrase"] == [project.wake_phrase]


def test_phrase_parts_use_runtime_overlay_and_generated_output_dirs(
    tmp_path: Path,
) -> None:
    """Phrase parts should route positive generation through WakeLab overlay."""
    project = _project(tmp_path)
    project.training_phrase_parts = ["Hey", "Orac"]
    project.inter_part_silence_min_ms = 80
    project.inter_part_silence_max_ms = 250
    project.piper_sample_generator_path.mkdir(parents=True)
    (project.piper_sample_generator_path / "generate_samples.py").write_text(
        "",
        encoding="utf-8",
    )

    job = build_training_job(project, "generate_clips")
    runtime_config = json.loads(
        (project.config_dir / "_runtime" / "generate_clips.json").read_text(
            encoding="utf-8"
        )
    )

    assert runtime_config["model_name"] == "hey_orac"
    assert runtime_config["output_dir"] == str(project.openwakeword_output_dir)
    assert runtime_config["piper_sample_generator_path"].endswith(
        "piper_sample_generator_overlay"
    )
    assert job.env["WAKELAB_TRAINING_PHRASE_PARTS_JSON"] == '["Hey", "Orac"]'
    assert job.env["WAKELAB_INTER_PART_SILENCE_MIN_MS"] == "80"
    assert job.env["WAKELAB_INTER_PART_SILENCE_MAX_MS"] == "250"


def test_generate_clips_block_message_points_to_project_tab(
    tmp_path: Path,
) -> None:
    """Generate Clips should tell the user where to configure the model."""
    project = _project(tmp_path)
    project.piper_voice_model_path = Path("")

    errors = training_tab._blocking_errors_for_stage(project, "generate_clips")

    assert any("Project tab field" in message for message in errors)
    assert any("Piper voice / generator model" in message for message in errors)


def test_build_training_job_augment_clips_uses_generated_clip_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Augment clips should run the openWakeWord augmentation phase."""
    project = _project(tmp_path)
    raw_dir = project.openwakeword_output_dir / project.model_name / "positive_train"
    raw_dir.mkdir(parents=True)
    (raw_dir / "0.wav").write_bytes(b"wav")
    (project.openwakeword_output_dir / project.model_name / "positive_test").mkdir(
        parents=True
    )
    (project.openwakeword_output_dir / project.model_name / "negative_train").mkdir(
        parents=True
    )
    (project.openwakeword_output_dir / project.model_name / "negative_test").mkdir(
        parents=True
    )
    (project.openwakeword_repo / "openwakeword" / "resources" / "models").mkdir(
        parents=True,
    )
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        (
            project.openwakeword_repo
            / "openwakeword"
            / "resources"
            / "models"
            / name
        ).write_bytes(b"onnx")
    ensured_repos = []
    monkeypatch.setattr(
        training_tab,
        "ensure_openwakeword_training_assets",
        lambda repo_path: ensured_repos.append(repo_path) or [],
    )
    job = build_training_job(project, "augment_clips")

    assert ensured_repos == [project.openwakeword_repo]
    assert job.command[:3] == [
        sys.executable,
        "-m",
        "orac_wake_lab.services.openwakeword_train_runner",
    ]
    assert "--augment_clips" in job.command
    assert "--overwrite" in job.command
    assert job.command[4].endswith("augment_clips.json")


def test_train_model_blocks_when_features_are_missing(tmp_path: Path) -> None:
    """Train Model should tell the user to run Augment Clips first."""
    project = _project(tmp_path)
    project.generated_clips_dir.mkdir(parents=True)
    for index in range(4):
        (project.generated_clips_dir / f"{index}.wav").write_bytes(b"wav")
    (project.openwakeword_repo / "openwakeword" / "resources" / "models").mkdir(
        parents=True,
    )
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        (
            project.openwakeword_repo
            / "openwakeword"
            / "resources"
            / "models"
            / name
        ).write_bytes(b"onnx")

    errors = training_tab._blocking_errors_for_stage(project, "train_model")

    assert any("Run Augment Clips first" in message for message in errors)


def test_clean_generated_training_outputs_removes_stale_clips_and_features(
    tmp_path: Path,
) -> None:
    """Regeneration cleanup should remove generated inputs, not model exports."""
    project = _project(tmp_path)
    model_dir = project.openwakeword_output_dir / project.model_name
    for directory_name in training_tab.GENERATED_CLIP_DIR_NAMES:
        clip_dir = model_dir / directory_name
        clip_dir.mkdir(parents=True)
        (clip_dir / "0.wav").write_bytes(b"wav")
    for file_name in training_tab.GENERATED_FEATURE_FILE_NAMES:
        (model_dir / file_name).write_bytes(b"features")
    exported_model = project.openwakeword_output_dir / f"{project.model_name}.onnx"
    exported_model.parent.mkdir(parents=True, exist_ok=True)
    exported_model.write_bytes(b"model")

    removed = training_tab.clean_generated_training_outputs(project)

    assert removed == (
        len(training_tab.GENERATED_CLIP_DIR_NAMES)
        + len(training_tab.GENERATED_FEATURE_FILE_NAMES)
    )
    for directory_name in training_tab.GENERATED_CLIP_DIR_NAMES:
        assert not (model_dir / directory_name).exists()
    for file_name in training_tab.GENERATED_FEATURE_FILE_NAMES:
        assert not (model_dir / file_name).exists()
    assert exported_model.read_bytes() == b"model"


def test_clean_generated_training_outputs_preserves_real_positives(
    tmp_path: Path,
) -> None:
    """Regeneration cleanup should not delete imported user recordings."""
    project = _project(tmp_path)
    create_project_workspace(project)
    real_clip = project.real_positive_clips_dir / "user_001.wav"
    wavfile.write(real_clip, 16000, np.full(16000, 1000, dtype=np.int16))
    staged_dir = (
        project.openwakeword_output_dir / project.model_name / "positive_train"
    )
    staged_dir.mkdir(parents=True)
    (staged_dir / "real_positive_0000.wav").write_bytes(b"staged")

    removed = training_tab.clean_generated_training_outputs(project)

    assert removed == 1
    assert real_clip.exists()
    assert not staged_dir.exists()


def test_resample_directory_to_16khz(tmp_path: Path) -> None:
    """Generated clips should be normalized to 16 kHz before augmenting."""
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    wav_path = clip_dir / "0.wav"
    wavfile.write(wav_path, 22050, np.zeros(22050, dtype=np.int16))

    resampled = resample_directory_to_16khz(clip_dir)

    rate, _ = wavfile.read(wav_path)
    assert resampled == 1
    assert rate == 16000


def test_phrase_parts_generate_separate_fragments_with_bounded_silence(
    tmp_path: Path,
) -> None:
    """Phrase-part generation should concatenate separate Piper fragments."""
    calls: list[str] = []

    def fake_generator(
        *,
        text: list[str],
        output_dir: Path,
        max_samples: int,
        file_names: list[str],
        **_kwargs: object,
    ) -> None:
        calls.append(text[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        amplitude = 1000 if text[0] == "Hey" else 2000
        frame_count = 80 if text[0] == "Hey" else 140
        for file_name in file_names[:max_samples]:
            wavfile.write(
                output_dir / file_name,
                1000,
                np.full(frame_count, amplitude, dtype=np.int16),
            )

    positive_clip_generator.generate_samples_with_word_boundaries(
        base_generator=fake_generator,
        text=["Hey Orac"],
        output_dir=tmp_path / "positive_train",
        phrase_parts=["Hey", "Orac"],
        silence_min_ms=80,
        silence_max_ms=120,
        max_samples=3,
        file_names=["a.wav", "b.wav", "c.wav"],
    )

    assert calls == ["Hey", "Orac"]
    for wav_path in sorted((tmp_path / "positive_train").glob("*.wav")):
        rate, data = wavfile.read(wav_path)
        inserted_silence_ms = len(data) - 220
        assert rate == 1000
        assert 80 <= inserted_silence_ms <= 120
        assert data[0] == 1000
        assert data[-1] == 2000


def test_piper_voice_test_uses_selected_model_and_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voice testing should synthesize and play the selected sample."""
    model_path = tmp_path / "voices" / "en_US.onnx"
    generator_path = tmp_path / "piper"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("model", encoding="utf-8")
    generator_path.mkdir(parents=True)
    played: list[Path] = []

    def fake_synthesise_sample(
        *,
        model_path: Path,
        generator_path: Path,
        output_dir: Path,
    ) -> None:
        assert model_path == model_path_arg
        assert generator_path == generator_path_arg
        (output_dir / "0.wav").write_bytes(b"wav")

    def fake_play_wav(wav_path: Path) -> None:
        played.append(wav_path)

    model_path_arg = model_path
    generator_path_arg = generator_path
    monkeypatch.setattr(
        piper_voice_test,
        "_synthesise_sample",
        fake_synthesise_sample,
    )
    monkeypatch.setattr(piper_voice_test, "_play_wav", fake_play_wav)

    piper_voice_test.test_piper_voice(model_path, generator_path)

    assert played
    assert played[0].name == "0.wav"


def test_directory_evaluation_labels_real_positives_separately(
    tmp_path: Path,
) -> None:
    """Directory evaluation output should preserve the evaluated group label."""
    project = _project(tmp_path)
    result = model_tester.WavTestResult(
        model_path=tmp_path / "model.onnx",
        wav_path=project.real_positive_clips_dir / "user_001.wav",
        model_name=project.model_name,
        threshold=0.75,
        max_score=0.91,
        frame_count=4,
    )

    rendered = model_tester.render_directory_results(
        "Real positive clips",
        project.real_positive_clips_dir,
        [result],
    )

    assert "Real positive clips" in rendered
    assert "user_001.wav" in rendered
    assert "Synthetic" not in rendered


def test_build_training_job_sets_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess environment should include Piper generator in PYTHONPATH."""
    project = _project(tmp_path)
    project.generated_clips_dir.mkdir(parents=True)
    for index in range(4):
        (project.generated_clips_dir / f"{index}.wav").write_bytes(b"wav")
    (project.openwakeword_repo / "openwakeword" / "resources" / "models").mkdir(
        parents=True,
    )
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        (
            project.openwakeword_repo
            / "openwakeword"
            / "resources"
            / "models"
            / name
        ).write_bytes(b"onnx")
    monkeypatch.setattr(
        training_tab,
        "ensure_openwakeword_training_assets",
        lambda _repo_path: [],
    )
    job = build_training_job(project, "train_model")

    assert "PYTHONPATH" in job.env
    assert str(project.piper_sample_generator_path) in job.env["PYTHONPATH"]
    assert job.env["CUDA_VISIBLE_DEVICES"] == ""
    assert job.env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert job.command[4].endswith("train_model.json")
    runtime_config_path = project.config_dir / "_runtime" / "train_model.json"
    overlay_module = (
        project.config_dir
        / "_runtime"
        / "piper_sample_generator_overlay"
        / "generate_samples.py"
    )
    assert runtime_config_path.exists()
    assert overlay_module.exists()
    overlay_text = overlay_module.read_text(
        encoding="utf-8"
    )
    assert "piper_sample_generator.__main__" in overlay_text
    assert "generate_samples_onnx" in overlay_text
    assert "WAKELAB_PIPER_MODEL_PATH" in overlay_text
    runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    assert runtime_config["model_name"] == project.model_name
    assert runtime_config["piper_sample_generator_path"] == str(
        project.config_dir
        / "_runtime"
        / "piper_sample_generator_overlay"
    )

    augment_job = build_training_job(project, "augment_clips")
    assert (
        augment_job.command[2]
        == "orac_wake_lab.services.openwakeword_train_runner"
    )
    assert augment_job.command[4].endswith("augment_clips.json")


def test_piper_voice_test_disables_cuda_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voice testing should force the generator onto CPU."""
    model_path = tmp_path / "voices" / "en_US.onnx"
    generator_path = tmp_path / "piper"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("model", encoding="utf-8")
    generator_path.mkdir(parents=True)
    captured_env: dict[str, str] = {}

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
        stdout: object,
        stderr: object,
        text: bool,
    ) -> object:
        captured_env.update(env)
        return object()

    monkeypatch.setattr(piper_voice_test.subprocess, "run", fake_run)
    monkeypatch.setattr(piper_voice_test.shutil, "which", lambda _name: "aplay")

    piper_voice_test._synthesise_sample(
        model_path=model_path,
        generator_path=generator_path,
        output_dir=tmp_path / "output",
    )

    assert captured_env["CUDA_VISIBLE_DEVICES"] == ""


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


def test_process_runner_filters_debug_lines(tmp_path: Path) -> None:
    """Runner logs should omit debug chatter from subprocesses."""
    runner = ProcessRunner(cancel_timeout_seconds=0.2)
    script = tmp_path / "emit_logs.py"
    script.write_text(
        "\n".join(
            [
                "print('DEBUG: noisy')",
                "print('INFO: useful')",
                "print('plain output')",
            ]
        ),
        encoding="utf-8",
    )
    job = TrainingJob(
        name="debug_filter",
        command=[
            sys.executable,
            str(script),
        ],
        cwd=tmp_path,
        log_path=tmp_path / "debug_filter.log",
    )
    emitted: list[str] = []

    runner.start(job, on_output=emitted.append)
    _wait_until(lambda: job.status != JobStatus.RUNNING, timeout=3.0)

    log_text = job.log_path.read_text(encoding="utf-8")
    output_lines = [line for line in log_text.splitlines() if line]
    assert all("DEBUG: noisy" not in line for line in output_lines)
    assert all("DEBUG: noisy" not in line for line in emitted)
    assert any("INFO: useful" in line for line in output_lines)
    assert any("plain output" in line for line in output_lines)


def test_training_tab_uses_stage_completion_banners(tmp_path: Path) -> None:
    """Train tab completion messages should include readable banners."""
    project = _project(tmp_path)
    tab = training_tab.TrainingTab.__new__(training_tab.TrainingTab)
    tab.app = _FakeApp()
    tab.app.project = project
    tab.run_all_queue = deque()
    tab.status_var = _FakeStatusVar()
    tab.log = _FakeChecksOutput("")
    appended: list[str] = []
    tab._append_log = lambda line: appended.append(line)  # type: ignore[method-assign]
    tab.clear_button = _FakeButton()

    completed_job = TrainingJob(
        name="generate_clips",
        command=[sys.executable],
        cwd=tmp_path,
        log_path=tmp_path / "job.log",
        status=JobStatus.COMPLETED,
    )
    failed_job = TrainingJob(
        name="train_model",
        command=[sys.executable],
        cwd=tmp_path,
        log_path=tmp_path / "job.log",
        status=JobStatus.FAILED,
    )

    training_tab.TrainingTab._job_completed(tab, completed_job)
    assert "Generate Clips: Completed" in appended[-1]
    assert tab.clear_button.state == "normal"

    training_tab.TrainingTab._job_completed(tab, failed_job)
    assert "Train Model: Failed" in appended[-1]
    assert tab.clear_button.state == "normal"


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


class _FakeStatusVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _FakeApp:
    def __init__(self) -> None:
        self.project: WakeWordProject | None = None

    def set_project(self, project: WakeWordProject) -> None:
        self.project = project

    def clear_project(self) -> None:
        self.project = None

    def get_project(self) -> WakeWordProject | None:
        return self.project


class _FakeChecksOutput:
    def __init__(self, text: str) -> None:
        self.text = text
        self.state = "normal"

    def get(self, start: str, end: str) -> str:
        return self.text

    def configure(self, **kwargs: object) -> None:
        if "state" in kwargs:
            self.state = str(kwargs["state"])

    def delete(self, start: str, end: str) -> None:
        self.text = ""

    def insert(self, index: str, text: str) -> None:
        self.text += text

    def see(self, index: str) -> None:
        return None


class _FakeChecksTab:
    def __init__(self) -> None:
        self.output_text = "[PASS] Python executable\n  /usr/bin/python\n"
        self.output = _FakeChecksOutput(self.output_text)
        self.clipboard_cleared = False
        self.clipboard_text = ""
        self.update_called = False

    def clipboard_clear(self) -> None:
        self.clipboard_cleared = True

    def clipboard_append(self, text: str) -> None:
        self.clipboard_text = text

    def update_idletasks(self) -> None:
        self.update_called = True


class _FakeClipboardTab:
    def __init__(self, text: str) -> None:
        self.log = _FakeChecksOutput(text)
        self.clipboard_cleared = False
        self.clipboard_text = ""
        self.update_called = False

    def clipboard_clear(self) -> None:
        self.clipboard_cleared = True

    def clipboard_append(self, text: str) -> None:
        self.clipboard_text = text

    def update_idletasks(self) -> None:
        self.update_called = True


class _FakeRunner:
    def __init__(self, *, is_running: bool) -> None:
        self.is_running = is_running


class _FakeButton:
    def __init__(self) -> None:
        self.state = "normal"

    def configure(self, **kwargs: object) -> None:
        if "state" in kwargs:
            self.state = str(kwargs["state"])


class _FakeProjectMenu:
    def __init__(self) -> None:
        self.values: list[str] = []

    def configure(self, **kwargs: object) -> None:
        values = kwargs.get("values")
        if isinstance(values, list):
            self.values = values


class _FakeTestTab:
    def __init__(self, project: WakeWordProject | None = None) -> None:
        self.app = _FakeApp()
        if project is not None:
            self.app.project = project
        self.model_path_var = _FakeMutablePathVar("")
        self.wav_path_var = _FakeMutablePathVar("")
        self.output_text = "test output"
        self.clipboard_cleared = False
        self.clipboard_text = ""
        self.update_called = False

    def clipboard_clear(self) -> None:
        self.clipboard_cleared = True

    def clipboard_append(self, text: str) -> None:
        self.clipboard_text = text

    def update_idletasks(self) -> None:
        self.update_called = True


class _FakeExportTab:
    def __init__(self, project: WakeWordProject | None = None) -> None:
        self.app = _FakeApp()
        if project is not None:
            self.app.project = project
        self.model_path_var = _FakeMutablePathVar("")


class _FakeLabel:
    def __init__(self) -> None:
        self.text = ""

    def configure(self, **kwargs: object) -> None:
        self.text = str(kwargs["text"])


class _FakeRefreshTab:
    def __init__(self) -> None:
        self.refreshed = False

    def refresh_from_project(self) -> None:
        self.refreshed = True


class _FakeProjectTab:
    def __init__(self) -> None:
        self.app = _FakeApp()
        self.status_var = _FakeStatusVar()
        self.applied_project: WakeWordProject | None = None

    def apply_project(self, project: WakeWordProject) -> None:
        self.applied_project = project


class _FakePathVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _FakeMutablePathVar(_FakePathVar):
    def set(self, value: str) -> None:
        self._value = value


class _FakeBoolVar:
    def __init__(self, value: bool) -> None:
        self._value = value

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = value


class _FakePiperModelTab:
    def __init__(self, project: WakeWordProject, model_path: Path) -> None:
        self.app = _FakeApp()
        self.app.project = project
        self.piper_model_path_var = _FakePathVar(str(model_path))
        self.status_var = _FakeStatusVar()


def _project(tmp_path: Path) -> WakeWordProject:
    return WakeWordProject(
        wake_phrase="Hey Orac",
        model_name="hey_orac",
        workspace_dir=tmp_path / "hey_orac",
        openwakeword_repo=tmp_path / "openwakeword",
        piper_sample_generator_path=tmp_path / "piper",
        piper_voice_model_path=tmp_path / "voices" / "en_US.onnx",
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
                "print('r' + 'eady', flush=True); "
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
