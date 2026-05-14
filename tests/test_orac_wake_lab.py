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
from orac_wake_lab.services.orac_export import build_orac_config_snippet
from orac_wake_lab.services.orac_export import build_smoke_test_command
from orac_wake_lab.services.orac_export import export_model_to_orac
from orac_wake_lab.services.orac_export import find_generated_models
from orac_wake_lab.services.process_runner import ProcessRunner
from orac_wake_lab.services import openwakeword_train_runner
from orac_wake_lab.services.project_store import create_project_workspace
from orac_wake_lab.services.project_store import discover_projects
from orac_wake_lab.services.project_store import load_project
from orac_wake_lab.services.project_store import project_dir_for_model
from orac_wake_lab.services import piper_voice_test
from orac_wake_lab.services.training_config import write_training_config
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


def test_saved_project_can_be_discovered_after_restart(
    tmp_path: Path,
) -> None:
    """Saved projects should be discovered from the projects root."""
    project = _project(tmp_path)
    create_project_workspace(project)

    discovered = discover_projects(tmp_path)

    assert [item.model_name for item in discovered] == ["hey_orac"]
    assert discovered[0].workspace_dir == project.workspace_dir


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


def test_test_tab_wav_browser_starts_in_project_clip_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAV browser should open near generated project clips by default."""
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

    assert captured["initialdir"] == str(
        project.openwakeword_output_dir / project.model_name
    )


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
    monkeypatch.setattr(
        training_tab,
        "ensure_openwakeword_training_assets",
        lambda _repo_path: [],
    )
    job = build_training_job(project, "augment_clips")

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


def test_training_tab_uses_short_completion_summary(tmp_path: Path) -> None:
    """Train tab completion messages should stay concise."""
    project = _project(tmp_path)
    tab = training_tab.TrainingTab.__new__(training_tab.TrainingTab)
    tab.app = _FakeApp()
    tab.app.project = project
    tab.run_all_queue = deque()
    tab.status_var = _FakeStatusVar()
    tab.log = _FakeChecksOutput("")
    tab._append_log = lambda line: setattr(tab, "_last_log", line)  # type: ignore[method-assign]

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
    assert tab._last_log == "generate_clips completed.\n"

    training_tab.TrainingTab._job_completed(tab, failed_job)
    assert tab._last_log == "train_model failed.\n"


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

    def get(self, start: str, end: str) -> str:
        return self.text


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
