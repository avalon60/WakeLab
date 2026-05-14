"""Training execution tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-13
# Description: Provides buttons and log output for training subprocesses.

from __future__ import annotations

import json
import importlib
import os
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
from orac_wake_lab.services.training_config import build_training_config
from orac_wake_lab.services.training_config import TRAINING_PROFILES
from orac_wake_lab.services.training_config import (
    validate_training_config_inputs,
)


def _path_is_unset(path: object) -> bool:
    """Return True when a field represents an intentionally blank path."""
    return str(path).strip() in {"", "."}


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
    if (project.piper_sample_generator_path / "generate_samples.py").exists():
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
            "        return _generate_samples_onnx(*args, **kwargs)\n"
            "    if model_suffix == '.pt':\n"
            "        return _generate_samples(*args, **kwargs)\n"
            "    raise ValueError(\n"
            "        'Unsupported Piper model type. Expected .onnx voice or '\n"
            "        f'.pt generator model: {model_value}'\n"
            "    )\n"
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
            "starting training."
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
            self._append_log("Create a project first.\n")
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
            job = build_training_job(project, stage)
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
            self._append_log(f"Unable to start {stage}: {exc}\n")

    def _start_next_queued_stage(self) -> None:
        if not self.run_all_queue:
            return
        self.start_stage(self.run_all_queue.popleft())

    def _job_completed(self, job: TrainingJob) -> None:
        if job.name == "generate_clips" and job.status == JobStatus.COMPLETED:
            resampled = _resample_generated_outputs(self.app.get_project())
            if resampled:
                self._append_log(
                    f"Resampled {resampled} generated clips to 16 kHz.\n"
                )
        self.status_var.set(f"{job.name}: {job.status.value}")
        if job.status == JobStatus.COMPLETED:
            self._append_log(f"{job.name} completed.\n")
        else:
            self._append_log(f"{job.name} failed.\n")
        if job.status == JobStatus.COMPLETED and self.run_all_queue:
            self._start_next_queued_stage()
        elif job.status != JobStatus.COMPLETED:
            self.run_all_queue.clear()

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
    elif stage == "augment_clips":
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
