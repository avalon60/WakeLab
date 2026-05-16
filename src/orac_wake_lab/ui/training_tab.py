"""Training execution tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-13
# Description: Provides buttons and log output for training subprocesses.

from __future__ import annotations

import json
import importlib
import os
import platform
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import customtkinter as ctk

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.training_job import JobStatus
from orac_wake_lab.models.training_job import TrainingJob
from orac_wake_lab.services.audio_resample import (
    resample_directory_to_16khz,
)
from orac_wake_lab.services.feature_bundle import STANDARD_FEATURE_BUNDLE_MESSAGE
from orac_wake_lab.services.feature_bundle import validate_feature_file
from orac_wake_lab.services.openwakeword_assets import (
    ensure_openwakeword_training_assets,
)
from orac_wake_lab.services.process_runner import ProcessRunner
from orac_wake_lab.services.real_positives import (
    stage_real_positives_for_training,
)
from orac_wake_lab.services.training_config import build_training_config
from orac_wake_lab.services.training_config import TRAINING_PROFILES
from orac_wake_lab.services.training_config import (
    validate_training_config_inputs,
)
from orac_wake_lab.services.training_config import (
    validate_positive_generation_settings,
)
from orac_wake_lab.services.training_config import validate_training_text_fields


GENERATED_CLIP_DIR_NAMES = (
    "positive_train",
    "positive_test",
    "negative_train",
    "negative_test",
)
GENERATED_FEATURE_FILE_NAMES = (
    "positive_features_train.npy",
    "positive_features_test.npy",
    "negative_features_train.npy",
    "negative_features_test.npy",
)
STAGE_LABELS = {
    "generate_clips": "Generate Clips",
    "augment_clips": "Augment Clips",
    "train_model": "Train Model",
    "train_convert_tflite": "Train + Convert TFLite",
}


def _path_is_unset(path: object) -> bool:
    """Return True when a field represents an intentionally blank path."""
    return str(path).strip() in {"", "."}


def _log_banner(title: str, detail: str) -> str:
    """Return a readable training log stage separator."""
    width = 72
    line = "=" * width
    text = f" {title}: {detail} "
    if len(text) >= width - 4:
        middle = f"--{text}--"
    else:
        remaining = width - len(text) - 4
        left = remaining // 2
        right = remaining - left
        middle = f"--{'-' * left}{text}{'-' * right}--"
    return f"\n{line}\n{middle}\n{line}\n"


def _generate_samples_compatibility_overlay(
    project: WakeWordProject,
) -> Path | None:
    """Create a runtime import overlay for ``generate_samples`` if needed.

    The current ``openWakeWord.train`` entrypoint imports a top-level
    ``generate_samples`` module from the configured Piper helper path. The
    checked-out Piper sample generator exposes the implementation through the
    ``piper_sample_generator`` package instead, so Wake Lab creates a small
    compatibility shim when the top-level module is absent.

    Args:
        project (WakeWordProject): Project settings.

    Returns:
        Path | None: Overlay directory path when created, otherwise ``None``.
    """
    needs_word_boundary_overlay = bool(
        project.training_phrase_parts
        or project.training_pronunciation_phrase.strip()
    )
    if (
        not needs_word_boundary_overlay
        and (project.piper_sample_generator_path / "generate_samples.py").exists()
    ):
        return None

    runtime_dir = project.config_dir / "_runtime"
    overlay_dir = runtime_dir / "piper_sample_generator_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / "generate_samples.py").write_text(
        (
            '"""Compatibility shim for openWakeWord train.py."""\n'
            "from __future__ import annotations\n\n"
            "import os\n\n"
            "from pathlib import Path\n\n"
            "import json\n\n"
            "from orac_wake_lab.services.positive_clip_generator import "
            "generate_samples_with_word_boundaries\n\n"
            "from piper_sample_generator.__main__ import generate_samples "
            "as _generate_samples\n"
            "from piper_sample_generator.__main__ import generate_samples_onnx "
            "as _generate_samples_onnx\n\n"
            "def generate_samples(*args, **kwargs):\n"
            "    model = kwargs.get('model')\n"
            "    if model is None:\n"
            "        model = os.environ.get('WAKELAB_PIPER_MODEL_PATH')\n"
            "    if model is None:\n"
            "        raise TypeError(\n"
            "            'generate_samples() missing required model path. '\n"
            "            'Set WAKELAB_PIPER_MODEL_PATH.'\n"
            "        )\n"
            "    kwargs['model'] = model\n"
            "    model_value = model[0] if isinstance(model, list) else model\n"
            "    model_suffix = Path(str(model_value)).suffix.lower()\n"
            "    if model_suffix == '.onnx':\n"
            "        generator = _generate_samples_onnx\n"
            "    elif model_suffix == '.pt':\n"
            "        generator = _generate_samples\n"
            "    else:\n"
            "        raise ValueError(\n"
            "            'Unsupported Piper model type. Expected .onnx voice or '\n"
            "            f'.pt generator model: {model_value}'\n"
            "    )\n"
            "    raw_output_dir = kwargs.get(\n"
            "        'output_dir', args[1] if len(args) > 1 else ''\n"
            "    )\n"
            "    output_dir = Path(str(raw_output_dir))\n"
            "    is_positive_dir = output_dir.name in {\n"
            "        'positive_train', 'positive_test'\n"
            "    }\n"
            "    phrase_parts = json.loads(\n"
            "        os.environ.get('WAKELAB_TRAINING_PHRASE_PARTS_JSON', '[]')\n"
            "    )\n"
            "    pronunciation_phrase = os.environ.get(\n"
            "        'WAKELAB_TRAINING_PRONUNCIATION_PHRASE', ''\n"
            "    ).strip()\n"
            "    if is_positive_dir and phrase_parts:\n"
            "        generation_kwargs = dict(kwargs)\n"
            "        text = generation_kwargs.pop(\n"
            "            'text', args[0] if len(args) > 0 else None\n"
            "        )\n"
            "        output_dir = generation_kwargs.pop(\n"
            "            'output_dir', args[1] if len(args) > 1 else None\n"
            "        )\n"
            "        max_samples = generation_kwargs.pop('max_samples', None)\n"
            "        file_names = generation_kwargs.pop('file_names', None)\n"
            "        return generate_samples_with_word_boundaries(\n"
            "            base_generator=generator,\n"
            "            text=text,\n"
            "            output_dir=output_dir,\n"
            "            phrase_parts=phrase_parts,\n"
            "            silence_min_ms=int(os.environ.get(\n"
            "                'WAKELAB_INTER_PART_SILENCE_MIN_MS', '80'\n"
            "            )),\n"
            "            silence_max_ms=int(os.environ.get(\n"
            "                'WAKELAB_INTER_PART_SILENCE_MAX_MS', '250'\n"
            "            )),\n"
            "            max_samples=max_samples,\n"
            "            file_names=file_names,\n"
            "            **generation_kwargs,\n"
            "        )\n"
            "    if is_positive_dir and pronunciation_phrase:\n"
            "        if args:\n"
            "            args = ([pronunciation_phrase], *args[1:])\n"
            "        else:\n"
            "            kwargs['text'] = [pronunciation_phrase]\n"
            "    return generator(*args, **kwargs)\n"
        ),
        encoding="utf-8",
    )
    return overlay_dir


def _runtime_training_config_path(
    project: WakeWordProject,
    stage: str,
) -> Path:
    """Write a stage-specific runtime config for ``openWakeWord.train``.

    Args:
        project (WakeWordProject): Project settings.
        stage (str): Stage identifier.

    Returns:
        Path: Path to the runtime training config.
    """
    runtime_dir = project.config_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config = build_training_config(project)
    overlay_dir = _generate_samples_compatibility_overlay(project)
    if overlay_dir is not None:
        config["piper_sample_generator_path"] = str(overlay_dir)
    runtime_config_path = runtime_dir / f"{stage}.json"
    runtime_config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    return runtime_config_path


class TrainingTab(ctk.CTkFrame):
    """Training subprocess tab."""

    def __init__(self, master: ctk.CTkBaseClass, app: object) -> None:
        """Create the training tab."""
        super().__init__(master)
        self.app = app
        self.runner = ProcessRunner()
        self.run_all_queue: deque[str] = deque()
        self.status_var = ctk.StringVar(value="Idle.")
        self._build()

    def _build(self) -> None:
        help_text = (
            "Generate Clips needs a Piper voice or generator model selected "
            "in the Project tab. Use a .onnx or .pt file. Run Checks first if "
            "you want the exact missing prerequisites called out before "
            "starting training. If the TTS merges words together, set phrase "
            "parts in Project and listen to generated positive clips before "
            "training. Generate Clips clears stale generated clips and "
            "derived generated feature arrays before regenerating."
        )
        ctk.CTkLabel(
            self,
            text=help_text,
            justify="left",
            wraplength=1020,
        ).pack(fill="x", padx=12, pady=(12, 0), anchor="w")

        buttons = ctk.CTkFrame(self)
        buttons.pack(fill="x", padx=12, pady=12)
        for label, stage in [
            ("Generate Clips", "generate_clips"),
            ("Augment Clips", "augment_clips"),
            ("Train Model", "train_model"),
            ("Train + Convert To TFLite", "train_convert_tflite"),
        ]:
            ctk.CTkButton(
                buttons,
                text=label,
                command=lambda s=stage: self.start_stage(s),
            ).pack(side="left", padx=4)
        ctk.CTkButton(
            buttons,
            text="Run All",
            command=self.run_all,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            buttons,
            text="Cancel",
            command=self.runner.cancel,
        ).pack(side="left", padx=4)
        self.clear_button = ctk.CTkButton(
            buttons,
            text="Clear",
            command=self.clear_log,
        )
        self.clear_button.pack(side="left", padx=4)
        ctk.CTkButton(
            buttons,
            text="Open Positive Clips",
            command=self.open_generated_positive_clips_folder,
        ).pack(side="left", padx=4)
        ctk.CTkLabel(self, textvariable=self.status_var).pack(
            anchor="w",
            padx=12,
        )
        self.log = ctk.CTkTextbox(self, wrap="none")
        self.log.pack(fill="both", expand=True, padx=12, pady=12)
        self.log.configure(state="disabled")
        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.pack(anchor="w", padx=12, pady=(0, 12))
        ctk.CTkButton(
            button_row,
            text="Copy To Clipboard",
            command=self.copy_log_to_clipboard,
        ).pack(side="left")

    def run_all(self) -> None:
        """Run all Phase 1 training stages in sequence."""
        self._append_log(_log_banner("Run All", "Starting training pipeline"))
        self.run_all_queue = deque(
            [
                "generate_clips",
                "augment_clips",
                "train_model",
            ]
        )
        self._start_next_queued_stage()

    def start_stage(self, stage: str) -> None:
        """Start one training stage.

        Args:
            stage (str): Stage identifier.
        """
        project = self.app.get_project()
        if project is None:
            self._append_log("Create or open a project first.\n")
            return
        blocking_errors = _blocking_errors_for_stage(project, stage)
        if blocking_errors:
            self._append_log(
                "Stage blocked by failed checks:\n"
                + "\n".join(f"  {message}" for message in blocking_errors)
                + "\n",
            )
            return
        try:
            self._append_log(
                _log_banner(
                    STAGE_LABELS.get(stage, stage),
                    "Preparing stage",
                )
            )
            if stage == "generate_clips":
                removed = clean_generated_training_outputs(project)
                if removed:
                    self._append_log(
                        "Cleared stale generated clips/features before "
                        f"regeneration: {removed} file(s).\n"
                    )
            elif stage == "augment_clips":
                staged = stage_real_positives_for_training(project)
                if staged:
                    self._append_log(
                        f"Staged {staged} real-positive training copies.\n"
                    )
            job = build_training_job(project, stage)
            self._append_log(
                _log_banner(
                    STAGE_LABELS.get(stage, stage),
                    "Running subprocess",
                )
            )
            self._set_clear_enabled(False)
            self.runner.start(
                job,
                on_output=lambda line: self.after(
                    0,
                    self._append_log,
                    line,
                ),
                on_complete=lambda completed: self.after(
                    0,
                    self._job_completed,
                    completed,
                ),
            )
            self.status_var.set(f"Running {stage}...")
        except Exception as exc:
            self._set_clear_enabled(True)
            self._append_log(f"Unable to start {stage}: {exc}\n")

    def _start_next_queued_stage(self) -> None:
        if not self.run_all_queue:
            return
        self.start_stage(self.run_all_queue.popleft())

    def _job_completed(self, job: TrainingJob) -> None:
        if job.name == "generate_clips" and job.status == JobStatus.COMPLETED:
            project = self.app.get_project()
            if project is None:
                self.run_all_queue.clear()
                self.status_var.set("project: missing")
                self._append_log("Project is no longer open.\n")
                self._set_clear_enabled(True)
                return
            resampled = _resample_generated_outputs(project)
            if resampled:
                self._append_log(
                    f"Resampled {resampled} generated clips to 16 kHz.\n"
                )
            try:
                staged = stage_real_positives_for_training(project)
            except Exception as exc:
                self.run_all_queue.clear()
                self.status_var.set("real positives: failed")
                self._append_log(f"Unable to stage real positives: {exc}\n")
                self._set_clear_enabled(True)
                return
            if staged:
                self._append_log(
                    f"Staged {staged} real-positive training copies.\n"
                )
        self.status_var.set(f"{job.name}: {job.status.value}")
        if job.status == JobStatus.COMPLETED:
            self._append_log(
                _log_banner(
                    STAGE_LABELS.get(job.name, job.name),
                    "Completed",
                )
            )
        else:
            self._append_log(
                _log_banner(
                    STAGE_LABELS.get(job.name, job.name),
                    "Failed",
                )
            )
        if job.status == JobStatus.COMPLETED and self.run_all_queue:
            self._start_next_queued_stage()
        elif job.status != JobStatus.COMPLETED:
            self.run_all_queue.clear()
            self._set_clear_enabled(True)
        else:
            self._set_clear_enabled(True)

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line)
        self.log.see("end")
        self.log.configure(state="disabled")

    def copy_log_to_clipboard(self) -> None:
        """Copy the current training log to the system clipboard."""
        text = self.log.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()

    def clear_log(self) -> None:
        """Clear the visible training log when no job is running."""
        if self.runner.is_running:
            return
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_clear_enabled(self, enabled: bool) -> None:
        """Enable or disable the training log clear button."""
        if hasattr(self, "clear_button"):
            self.clear_button.configure(
                state="normal" if enabled else "disabled"
            )

    def open_generated_positive_clips_folder(self) -> None:
        """Open the generated positive training clips folder."""
        project = self.app.get_project()
        if project is None:
            self._append_log("Create or open a project first.\n")
            return
        folder = (
            project.openwakeword_output_dir
            / project.model_name
            / "positive_train"
        )
        folder.mkdir(parents=True, exist_ok=True)
        try:
            _open_folder(folder)
        except Exception as exc:
            self._append_log(f"Unable to open generated clips folder: {exc}\n")
            return
        self._append_log(f"Generated positive clips folder: {folder}\n")


def build_training_job(project: WakeWordProject, stage: str) -> TrainingJob:
    """Build a subprocess job for an openWakeWord or Piper training stage.

    Args:
        project (WakeWordProject): Project settings.
        stage (str): Stage identifier.

    Returns:
        TrainingJob: Configured training job.
    """
    profile = TRAINING_PROFILES.get(
        project.profile,
        TRAINING_PROFILES["quick"],
    )
    supported_stages = {
        "generate_clips",
        "augment_clips",
        "train_model",
        "train_convert_tflite",
    }
    openwakeword_stage_args = {
        "generate_clips": [
            "--training_config",
            str(_runtime_training_config_path(project, "generate_clips")),
            "--generate_clips",
        ],
        "augment_clips": [
            "--training_config",
            str(_runtime_training_config_path(project, "augment_clips")),
            "--augment_clips",
            "--overwrite",
        ],
        "train_model": [
            "--training_config",
            str(_runtime_training_config_path(project, "train_model")),
            "--train_model",
        ],
        "train_convert_tflite": [
            "--training_config",
            str(_runtime_training_config_path(
                project,
                "train_convert_tflite",
            )),
            "--train_model",
            "--convert_to_tflite",
        ],
    }
    log_names = {
        "generate_clips": "generate_clips.log",
        "augment_clips": "augment_clips.log",
        "train_model": "train_model.log",
        "train_convert_tflite": "train_convert_tflite.log",
    }
    if stage not in supported_stages:
        raise ValueError(f"Unsupported training stage: {stage}")

    env = dict(os.environ)
    python_path = env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    
    if stage == "generate_clips":
        command = [
            sys.executable,
            "-m",
            "orac_wake_lab.services.openwakeword_train_runner",
            *openwakeword_stage_args[stage],
        ]
        cwd = project.openwakeword_repo
        if python_path:
            python_path = f"{project.piper_sample_generator_path}:{python_path}"
        else:
            python_path = str(project.piper_sample_generator_path)
        env["WAKELAB_PIPER_MODEL_PATH"] = str(project.piper_voice_model_path)
        _apply_positive_generation_env(project, env)
    elif stage == "augment_clips":
        ensure_openwakeword_training_assets(project.openwakeword_repo)
        command = [
            sys.executable,
            "-m",
            "orac_wake_lab.services.openwakeword_train_runner",
            *openwakeword_stage_args[stage],
        ]
        cwd = project.openwakeword_repo
        if python_path:
            python_path = f"{project.piper_sample_generator_path}:{python_path}"
        else:
            python_path = str(project.piper_sample_generator_path)
    else:
        ensure_openwakeword_training_assets(project.openwakeword_repo)
        command = [
            sys.executable,
            "-m",
            "orac_wake_lab.services.openwakeword_train_runner",
            *openwakeword_stage_args[stage],
        ]
        cwd = project.openwakeword_repo
        # Add Piper generator to PYTHONPATH for openwakeword.train if needed.
        if python_path:
            python_path = f"{project.piper_sample_generator_path}:{python_path}"
        else:
            python_path = str(project.piper_sample_generator_path)
        env["WAKELAB_PIPER_MODEL_PATH"] = str(project.piper_voice_model_path)
        _apply_positive_generation_env(project, env)

    if python_path:
        env["PYTHONPATH"] = python_path

    return TrainingJob(
        name=stage,
        command=command,
        cwd=cwd,
        log_path=project.logs_dir / log_names[stage],
        env=env,
    )


def _blocking_errors_for_stage(
    project: WakeWordProject,
    stage: str,
) -> list[str]:
    stage_blocks = {
        "generate_clips": {"generate"},
        "augment_clips": {"augment"},
        "train_model": {"train"},
        "train_convert_tflite": {"train", "convert_tflite"},
    }[stage]
    errors: list[str] = []
    if not project.openwakeword_repo.exists():
        errors.append(
            f"openWakeWord repository is missing: {project.openwakeword_repo}"
        )
    if "generate" in stage_blocks:
        text_result = validate_training_text_fields(project)
        if text_result.is_failure:
            errors.append(f"{text_result.name}: {text_result.message}")
        positive_generation_result = validate_positive_generation_settings(
            project
        )
        if positive_generation_result.is_failure:
            errors.append(
                f"{positive_generation_result.name}: "
                f"{positive_generation_result.message}"
            )
        if _path_is_unset(project.piper_voice_model_path):
            errors.append(
                "No Piper voice or generator model is configured. "
                "Set Project tab field 'Piper voice / generator model' "
                "to an existing .onnx or .pt file."
            )
        elif not project.piper_voice_model_path.exists():
            errors.append(
                "Piper voice or generator model is missing: "
                f"{project.piper_voice_model_path}"
            )
        elif project.piper_voice_model_path.suffix not in {".onnx", ".pt"}:
            errors.append(
                "Piper voice or generator model must be a .onnx or .pt file: "
                f"{project.piper_voice_model_path}"
            )

    if stage_blocks.intersection({"generate", "augment", "train"}):
        main_py = (
            project.piper_sample_generator_path
            / "piper_sample_generator"
            / "__main__.py"
        )
        if not main_py.exists():
            errors.append(
                "Piper sample generator is missing or has invalid layout: "
                f"{project.piper_sample_generator_path}"
            )
        if not project.background_paths:
            errors.append("No background directories are configured.")
        if not project.rir_paths:
            errors.append("No RIR directories are configured.")
        for path in project.background_paths or []:
            if not path.exists():
                errors.append(f"Background directory is missing: {path}")
        for path in project.rir_paths or []:
            if not path.exists():
                errors.append(f"RIR directory is missing: {path}")
    if "augment" in stage_blocks:
        try:
            importlib.import_module("audiomentations")
        except Exception as exc:
            errors.append(
                "Augment Clips requires audiomentations in the active "
                f"environment: {exc}"
            )
        try:
            importlib.import_module("torchcodec")
        except Exception as exc:
            errors.append(
                "Augment Clips requires torchcodec in the active "
                f"environment: {exc}"
            )
        required_dirs = {
            "positive_train": project.openwakeword_output_dir
            / project.model_name
            / "positive_train",
            "positive_test": project.openwakeword_output_dir
            / project.model_name
            / "positive_test",
            "negative_train": project.openwakeword_output_dir
            / project.model_name
            / "negative_train",
            "negative_test": project.openwakeword_output_dir
            / project.model_name
            / "negative_test",
        }
        missing_dirs = [
            f"{name}: {path}"
            for name, path in required_dirs.items()
            if not path.exists() or not list(path.glob("*.wav"))
        ]
        if missing_dirs:
            errors.append(
                "Run Generate Clips first to create the raw openWakeWord "
                "clip directories: " + ", ".join(missing_dirs)
            )
    if "train" in stage_blocks:
        validation_result = validate_feature_file(
            project.false_positive_validation_data_path,
            "False-positive validation feature file",
        )
        if validation_result.is_failure:
            errors.append(
                f"{STANDARD_FEATURE_BUNDLE_MESSAGE} {validation_result.message}"
            )
        training_result = validate_training_config_inputs(project)
        if training_result.is_failure:
            errors.append(
                f"{training_result.name}: {training_result.message}"
            )
        feature_dir = project.openwakeword_output_dir / project.model_name
        required_feature_files = [
            feature_dir / "positive_features_train.npy",
            feature_dir / "positive_features_test.npy",
            feature_dir / "negative_features_train.npy",
            feature_dir / "negative_features_test.npy",
        ]
        missing_features = [
            str(path) for path in required_feature_files if not path.exists()
        ]
        if missing_features:
            errors.append(
                "Run Augment Clips first to generate the openWakeWord "
                "feature files: " + ", ".join(missing_features)
            )
    return errors


def _apply_positive_generation_env(
    project: WakeWordProject,
    env: dict[str, str],
) -> None:
    """Add positive sample generation controls to a subprocess environment."""
    env["WAKELAB_TRAINING_PRONUNCIATION_PHRASE"] = (
        project.training_pronunciation_phrase
    )
    env["WAKELAB_TRAINING_PHRASE_PARTS_JSON"] = json.dumps(
        project.training_phrase_parts or []
    )
    env["WAKELAB_INTER_PART_SILENCE_MIN_MS"] = str(
        project.inter_part_silence_min_ms
    )
    env["WAKELAB_INTER_PART_SILENCE_MAX_MS"] = str(
        project.inter_part_silence_max_ms
    )


def _open_folder(folder: Path) -> None:
    """Open a folder in the platform file browser."""
    if os.name == "nt":
        os.startfile(folder)  # type: ignore[attr-defined]
        return
    if platform.system() == "Darwin":
        subprocess.Popen(["open", str(folder)])
        return
    subprocess.Popen(["xdg-open", str(folder)])


def clean_generated_training_outputs(project: WakeWordProject) -> int:
    """Delete generated clips and feature arrays for the active model.

    Args:
        project (WakeWordProject): Project settings.

    Returns:
        int: Number of files removed.
    """
    model_dir = project.openwakeword_output_dir / project.model_name
    removed_count = 0
    for directory_name in GENERATED_CLIP_DIR_NAMES:
        directory = model_dir / directory_name
        _ensure_project_output_child(model_dir, directory)
        if directory.exists():
            removed_count += len(
                [path for path in directory.rglob("*") if path.is_file()]
            )
            shutil.rmtree(directory)
    for file_name in GENERATED_FEATURE_FILE_NAMES:
        feature_file = model_dir / file_name
        _ensure_project_output_child(model_dir, feature_file)
        if feature_file.exists():
            feature_file.unlink()
            removed_count += 1
    return removed_count


def _ensure_project_output_child(model_dir: Path, path: Path) -> None:
    """Ensure a generated path is scoped to the model output directory."""
    resolved_model_dir = model_dir.resolve()
    resolved_path = path.resolve(strict=False)
    if resolved_model_dir not in resolved_path.parents:
        raise ValueError(f"Refusing to delete path outside model output: {path}")


def _resample_generated_outputs(project: WakeWordProject) -> int:
    """Normalize upstream-generated openWakeWord clips to 16 kHz.

    Args:
        project (WakeWordProject): Project settings.

    Returns:
        int: Number of WAV files resampled across all raw clip directories.
    """
    model_dir = project.openwakeword_output_dir / project.model_name
    count = 0
    for subdir in (
        "positive_train",
        "positive_test",
        "negative_train",
        "negative_test",
    ):
        count += resample_directory_to_16khz(model_dir / subdir)
    return count
