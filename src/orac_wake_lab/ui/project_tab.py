"""Project setup tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Builds the wake phrase and workspace setup UI.

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from orac_wake_lab.models.project import DEFAULT_OPENWAKEWORD_REPO
from orac_wake_lab.models.project import DEFAULT_ORAC_REPO
from orac_wake_lab.models.project import DEFAULT_WORKSPACE_ROOT
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import derive_model_name
from orac_wake_lab.models.validation import validate_model_name
from orac_wake_lab.models.validation import validate_phrase
from orac_wake_lab.services import wake_lab_home
from orac_wake_lab.services.project_store import create_project_workspace
from orac_wake_lab.services.project_store import project_dir_for_model
from orac_wake_lab.services.training_config import write_training_config


class ProjectTab(ctk.CTkFrame):
    """Project creation tab."""

    def __init__(self, master: ctk.CTkBaseClass, app: object) -> None:
        """Create the project tab.

        Args:
            master (ctk.CTkBaseClass): Parent widget.
            app (object): Main window controller.
        """
        super().__init__(master)
        self.app = app
        self.wake_phrase_var = ctk.StringVar(value="Hey Orac")
        self.model_name_var = ctk.StringVar(value="hey_orac")
        self.workspace_root_var = ctk.StringVar(
            value=str(DEFAULT_WORKSPACE_ROOT)
        )

        # Use managed defaults for paths
        self.piper_path_var = ctk.StringVar(
            value=str(wake_lab_home.detect_piper_sample_generator_path())
        )
        self.background_paths_var = ctk.StringVar(
            value=str(wake_lab_home.get_background_audio_dir())
        )
        self.rir_paths_var = ctk.StringVar(
            value=str(wake_lab_home.get_rir_dir())
        )

        # Discover precomputed features
        neg_features = wake_lab_home.discover_negative_features()
        self.negative_feature_paths_var = ctk.StringVar(
            value=",".join(str(p) for p in neg_features.values())
        )

        val_features = wake_lab_home.discover_validation_features()
        val_path = ""
        if len(val_features) == 1:
            val_path = str(val_features[0])
        self.validation_path_var = ctk.StringVar(value=val_path)

        self.negatives_var = ctk.StringVar(value="hey oracle, oracle, orac")
        self.profile_var = ctk.StringVar(value="quick")
        self.status_var = ctk.StringVar(value="No project created.")
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        rows = [
            ("Wake phrase", self.wake_phrase_var),
            ("Model name", self.model_name_var),
            ("Workspace root", self.workspace_root_var),
            ("Piper sample generator location", self.piper_path_var),
            ("Background audio dirs", self.background_paths_var),
            ("RIR dirs", self.rir_paths_var),
            ("Negative feature .npy files", self.negative_feature_paths_var),
            ("False-positive validation .npy", self.validation_path_var),
            ("Negative phrases", self.negatives_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ctk.CTkLabel(self, text=label).grid(
                row=row,
                column=0,
                padx=12,
                pady=6,
                sticky="w",
            )
            ctk.CTkEntry(self, textvariable=variable).grid(
                row=row,
                column=1,
                padx=12,
                pady=6,
                sticky="ew",
            )
            if label == "Model name":
                ctk.CTkButton(
                    self,
                    text="Derive",
                    width=80,
                    command=self._derive_model_name,
                ).grid(row=row, column=2, padx=12, pady=6)
            if (
                "location" in label.lower()
                or "dirs" in label.lower()
                or ".npy" in label.lower()
            ):
                ctk.CTkButton(
                    self,
                    text="Browse",
                    width=80,
                    command=lambda v=variable, l=label: self._browse_path(v, l),
                ).grid(row=row, column=2, padx=12, pady=6)

        self.wake_phrase_var.trace_add("write", self._phrase_changed)

        # Helper text for Piper
        ctk.CTkLabel(
            self,
            text=(
                "Used by openWakeWord to generate synthetic spoken examples "
                "of the wake phrase."
            ),
            font=("Arial", 10, "italic"),
        ).grid(row=3, column=1, sticky="w", padx=12)

        ctk.CTkLabel(self, text="Profile").grid(
            row=10,
            column=0,
            padx=12,
            pady=6,
            sticky="w",
        )
        ctk.CTkOptionMenu(
            self,
            variable=self.profile_var,
            values=["quick", "balanced", "manual"],
        ).grid(row=10, column=1, padx=12, pady=6, sticky="w")

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=11, column=1, padx=12, pady=12, sticky="w")

        ctk.CTkButton(
            button_frame,
            text="Create / Save Project",
            command=self.create_project,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            button_frame,
            text="Initialise Wake Lab Folders",
            command=self._initialize_folders,
            fg_color="gray30",
            hover_color="gray40",
        ).pack(side="left")

        ctk.CTkLabel(self, textvariable=self.status_var).grid(
            row=12,
            column=0,
            columnspan=3,
            padx=12,
            pady=8,
            sticky="w",
        )

    def _phrase_changed(self, *_args: object) -> None:
        if not self.model_name_var.get().strip():
            self._derive_model_name()

    def _derive_model_name(self) -> None:
        self.model_name_var.set(derive_model_name(self.wake_phrase_var.get()))

    def _initialize_folders(self) -> None:
        """Create the managed directory structure."""
        dirs = wake_lab_home.initialize_wake_lab_folders()
        home = wake_lab_home.get_wake_lab_home()
        self.status_var.set(f"Initialised {len(dirs)} folders under {home}")

    def _browse_path(self, variable: ctk.StringVar, label: str) -> None:
        initial_dir = None
        if "background" in label.lower():
            initial_dir = wake_lab_home.get_background_audio_dir()
        elif "rir" in label.lower():
            initial_dir = wake_lab_home.get_rir_dir()
        elif "negative feature" in label.lower():
            initial_dir = wake_lab_home.get_negative_features_dir()
        elif "false-positive" in label.lower():
            initial_dir = wake_lab_home.get_false_positive_validation_dir()
        elif "piper" in label.lower():
            initial_dir = wake_lab_home.detect_piper_sample_generator_path()

        if initial_dir and not initial_dir.exists():
            initial_dir = wake_lab_home.get_wake_lab_home()

        if variable is self.negative_feature_paths_var:
            selected_files = filedialog.askopenfilenames(
                initialdir=initial_dir,
                filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")]
            )
            selected = ",".join(selected_files)
        elif variable is self.validation_path_var:
            selected = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")]
            )
        else:
            selected = filedialog.askdirectory(initialdir=initial_dir)
        if selected:
            variable.set(selected)

    def create_project(self) -> None:
        """Create or update the current wake-word project."""
        phrase = self.wake_phrase_var.get().strip()
        model_name = self.model_name_var.get().strip()
        phrase_result = validate_phrase(phrase)
        model_result = validate_model_name(model_name)
        if phrase_result.is_failure or model_result.is_failure:
            self.status_var.set(
                f"{phrase_result.message} {model_result.message}"
            )
            return

        workspace_dir = project_dir_for_model(
            model_name,
            Path(self.workspace_root_var.get()),
        )
        project = WakeWordProject(
            wake_phrase=phrase,
            model_name=model_name,
            workspace_dir=workspace_dir,
            openwakeword_repo=DEFAULT_OPENWAKEWORD_REPO,
            orac_repo=DEFAULT_ORAC_REPO,
            piper_sample_generator_path=Path(self.piper_path_var.get()),
            background_paths=_split_paths(self.background_paths_var.get()),
            rir_paths=_split_paths(self.rir_paths_var.get()),
            negative_feature_data_files=_split_feature_files(
                self.negative_feature_paths_var.get()
            ),
            false_positive_validation_data_path=Path(
                self.validation_path_var.get()
            ),
            custom_negative_phrases=_split_values(self.negatives_var.get()),
            profile=self.profile_var.get(),
        )
        create_project_workspace(project)
        write_training_config(project)
        self.app.set_project(project)
        warning = (
            f" Warning: {phrase_result.message}"
            if phrase_result.status == "warn"
            else ""
        )
        self.status_var.set(f"Project saved: {workspace_dir}.{warning}")


def _split_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_feature_files(value: str) -> dict[str, Path]:
    paths = _split_paths(value)
    return {path.stem: path for path in paths}
