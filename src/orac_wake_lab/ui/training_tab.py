"""Training execution tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Provides buttons and log output for training subprocesses.

from __future__ import annotations

import os
import sys
from collections import deque

import customtkinter as ctk

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.training_job import JobStatus
from orac_wake_lab.models.training_job import TrainingJob
from orac_wake_lab.services.orac_export import find_generated_models
from orac_wake_lab.services.process_runner import ProcessRunner
from orac_wake_lab.services.training_config import (
    validate_training_config_inputs,
)


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

    def run_all(self) -> None:
        """Run all Phase 1 training stages in sequence."""
        self.run_all_queue = deque(
            [
                "generate_clips",
                "augment_clips",
                "train_convert_tflite",
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
        self.status_var.set(f"{job.name}: {job.status.value}")
        self._append_log(f"\n{job.name} finished: {job.status.value}\n")
        project = self.app.get_project()
        if project is not None:
            found = find_generated_models(project)
            if found:
                self._append_log(
                    "Mirrored models:\n"
                    + "\n".join(f"  {path}" for path in found)
                    + "\n",
                )
        if job.status == JobStatus.COMPLETED and self.run_all_queue:
            self._start_next_queued_stage()
        elif job.status != JobStatus.COMPLETED:
            self.run_all_queue.clear()

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line)
        self.log.see("end")
        self.log.configure(state="disabled")


def build_training_job(project: WakeWordProject, stage: str) -> TrainingJob:
    """Build a subprocess job for an openWakeWord or Piper training stage.

    Args:
        project (WakeWordProject): Project settings.
        stage (str): Stage identifier.

    Returns:
        TrainingJob: Configured training job.
    """
    stage_args = {
        "generate_clips": ["--training_config", str(project.training_config_path)],
        "augment_clips": ["--training_config", str(project.training_config_path)],
        "train_model": [
            "--training_config",
            str(project.training_config_path),
            "--train_model",
        ],
        "train_convert_tflite": [
            "--training_config",
            str(project.training_config_path),
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
    if stage not in stage_args:
        raise ValueError(f"Unsupported training stage: {stage}")

    env = dict(os.environ)
    python_path = env.get("PYTHONPATH", "")
    
    if stage == "generate_clips":
        command = [
            sys.executable,
            "-m",
            "piper_sample_generator",
            *stage_args[stage],
        ]
        cwd = project.piper_sample_generator_path
    elif stage == "augment_clips":
        command = [
            sys.executable,
            "-m",
            "piper_sample_generator.augment",
            *stage_args[stage],
        ]
        cwd = project.piper_sample_generator_path
    else:
        command = [
            sys.executable,
            "-m",
            "openwakeword.train",
            *stage_args[stage],
        ]
        cwd = project.openwakeword_repo
        # Add Piper generator to PYTHONPATH for openwakeword.train if needed
        if python_path:
            python_path = f"{project.piper_sample_generator_path}:{python_path}"
        else:
            python_path = str(project.piper_sample_generator_path)

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
    if "train" in stage_blocks:
        if not (
            project.false_positive_validation_data_path.exists()
            and project.false_positive_validation_data_path.suffix == ".npy"
        ):
            errors.append(
                "False-positive validation data is missing: "
                f"{project.false_positive_validation_data_path}"
            )
        training_result = validate_training_config_inputs(project)
        if training_result.is_failure:
            errors.append(
                f"{training_result.name}: {training_result.message}"
            )
    return errors
