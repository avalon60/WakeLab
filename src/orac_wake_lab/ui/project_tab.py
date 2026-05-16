"""Project setup tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Builds the wake phrase and workspace setup UI.

from __future__ import annotations

import os
import platform
import subprocess
import threading
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox

import customtkinter as ctk

from orac_wake_lab.models.project import DEFAULT_OPENWAKEWORD_REPO
from orac_wake_lab.models.project import DEFAULT_ORAC_REPO
from orac_wake_lab.models.project import DEFAULT_WORKSPACE_ROOT
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import derive_model_name
from orac_wake_lab.models.validation import validate_model_name
from orac_wake_lab.models.validation import validate_phrase
from orac_wake_lab.services import feature_bundle
from orac_wake_lab.services import piper_voice_test
from orac_wake_lab.services import real_positives
from orac_wake_lab.services import wake_lab_home
from orac_wake_lab.services.project_store import create_project_workspace
from orac_wake_lab.services.project_store import delete_project_workspace
from orac_wake_lab.services.project_store import discover_projects
from orac_wake_lab.services.project_store import load_project
from orac_wake_lab.services.project_store import project_dir_for_model
from orac_wake_lab.services.project_store import save_project
from orac_wake_lab.services.training_config import (
    validate_positive_generation_settings,
)
from orac_wake_lab.services.training_config import validate_training_text_fields
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
        defaults = _default_project_form_values()
        self.wake_phrase_var = ctk.StringVar(value=defaults["wake_phrase"])
        self.model_name_var = ctk.StringVar(value=defaults["model_name"])
        self.workspace_root_var = ctk.StringVar(
            value=defaults["workspace_root"]
        )

        # Use managed defaults for paths
        self.oww_path_var = ctk.StringVar(
            value=defaults["openwakeword_repo"]
        )
        self.orac_path_var = ctk.StringVar(
            value=defaults["orac_repo"]
        )
        self.piper_path_var = ctk.StringVar(
            value=defaults["piper_sample_generator_path"]
        )
        self.piper_model_path_var = ctk.StringVar(
            value=defaults["piper_voice_model_path"]
        )
        self.training_pronunciation_phrase_var = ctk.StringVar(
            value=defaults["training_pronunciation_phrase"]
        )
        self.training_phrase_parts_var = ctk.StringVar(
            value=defaults["training_phrase_parts"]
        )
        self.inter_part_silence_min_var = ctk.StringVar(
            value=defaults["inter_part_silence_min_ms"]
        )
        self.inter_part_silence_max_var = ctk.StringVar(
            value=defaults["inter_part_silence_max_ms"]
        )
        self.enable_real_positives_var = ctk.BooleanVar(
            value=defaults["enable_real_positives"] == "true"
        )
        self.real_positive_dir_var = ctk.StringVar(
            value=defaults["real_positive_clips_dir"]
        )
        self.real_positive_min_count_var = ctk.StringVar(
            value=defaults["real_positive_min_count"]
        )
        self.real_positive_target_percent_var = ctk.StringVar(
            value=defaults["real_positive_target_percent"]
        )
        self.background_paths_var = ctk.StringVar(
            value=defaults["background_paths"]
        )
        self.rir_paths_var = ctk.StringVar(
            value=defaults["rir_paths"]
        )

        self.negative_feature_paths_var = ctk.StringVar(
            value=defaults["negative_feature_data_files"]
        )
        self.validation_path_var = ctk.StringVar(
            value=defaults["false_positive_validation_data_path"]
        )

        self.negatives_var = ctk.StringVar(
            value=defaults["custom_negative_phrases"]
        )
        self.profile_var = ctk.StringVar(value=defaults["profile"])
        self.status_var = ctk.StringVar(value="No project created.")
        self.feature_bundle_status_var = ctk.StringVar(value="")
        self.selected_project_var = ctk.StringVar(value="")
        self.available_project_paths: dict[str, Path] = {}
        self._build()
        self.refresh_feature_bundle(auto_apply=True)

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_workflow_strip()

        self.content_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
        )
        self.content_scroll.grid(row=2, column=0, sticky="nsew")

        content = ctk.CTkFrame(
            self.content_scroll,
            fg_color="transparent",
            width=1040,
        )
        content.pack(anchor="n", pady=(12, 4))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        setup_section = self._add_section(
            content,
            row=0,
            column=0,
            title="1. Project Setup",
            subtitle="Name the wake-word project and choose the training profile.",
        )
        self._add_entry_row(
            setup_section,
            row=0,
            label="Wake phrase",
            variable=self.wake_phrase_var,
        )
        self._add_entry_row(
            setup_section,
            row=1,
            label="Model name",
            variable=self.model_name_var,
            buttons=[
                ("Derive", self._derive_model_name, 72),
            ],
        )
        self._add_option_row(
            setup_section,
            row=2,
            label="Profile",
            variable=self.profile_var,
            values=["quick", "balanced", "manual"],
        )

        voice_section = self._add_section(
            content,
            row=0,
            column=1,
            title="2. Voice Model",
            subtitle="Select the generator and voice model used for sample creation.",
        )
        self._add_entry_row(
            voice_section,
            row=0,
            label="Piper generator",
            variable=self.piper_path_var,
            browse=True,
        )
        self._add_entry_row(
            voice_section,
            row=1,
            label="Voice / generator model",
            variable=self.piper_model_path_var,
            browse=True,
            buttons=[
                ("Test", self.test_piper_voice_model, 62),
            ],
        )

        pronunciation_section = self._add_section(
            content,
            row=1,
            column=0,
            columnspan=2,
            title="3. Positive Sample Pronunciation",
            subtitle="Generate phrase parts separately when TTS merges word boundaries.",
        )
        self._add_entry_row(
            pronunciation_section,
            row=0,
            label="Training pronunciation phrase",
            variable=self.training_pronunciation_phrase_var,
        )
        self._add_entry_row(
            pronunciation_section,
            row=1,
            label="Training phrase parts",
            variable=self.training_phrase_parts_var,
        )
        self._add_entry_row(
            pronunciation_section,
            row=2,
            label="Inter-part silence min ms",
            variable=self.inter_part_silence_min_var,
        )
        self._add_entry_row(
            pronunciation_section,
            row=3,
            label="Inter-part silence max ms",
            variable=self.inter_part_silence_max_var,
        )

        data_section = self._add_section(
            content,
            row=2,
            column=0,
            columnspan=2,
            title="4. Training Data",
            subtitle="Manage augmentation audio, feature bundles, and negative phrases.",
        )
        self._add_entry_row(
            data_section,
            row=0,
            label="Background audio dirs",
            variable=self.background_paths_var,
            browse=True,
        )
        self._add_entry_row(
            data_section,
            row=1,
            label="RIR dirs",
            variable=self.rir_paths_var,
            browse=True,
        )
        self._add_entry_row(
            data_section,
            row=2,
            label="Negative feature .npy files",
            variable=self.negative_feature_paths_var,
            browse=True,
        )
        self._add_entry_row(
            data_section,
            row=3,
            label="False-positive validation .npy",
            variable=self.validation_path_var,
            browse=True,
        )
        self._add_entry_row(
            data_section,
            row=4,
            label="Negative phrases (near-miss sources)",
            variable=self.negatives_var,
        )
        self._add_real_positives_panel(data_section, row=5)
        self._add_feature_bundle_panel(data_section, row=6)

        advanced_section = self._add_collapsible_section(
            content,
            row=3,
            title="Advanced Infrastructure Settings",
            subtitle="Managed WakeLab paths are used by default. Change these only when you need custom checkouts or workspace locations.",
        )
        self._add_entry_row(
            advanced_section,
            row=0,
            label="Workspace root",
            variable=self.workspace_root_var,
            browse=True,
        )
        self._add_entry_row(
            advanced_section,
            row=1,
            label="openWakeWord checkout",
            variable=self.oww_path_var,
            browse=True,
        )
        self._add_entry_row(
            advanced_section,
            row=2,
            label="Orac checkout",
            variable=self.orac_path_var,
            browse=True,
        )

        self._build_footer(content, row=4)

        self.wake_phrase_var.trace_add("write", self._phrase_changed)
        self.refresh_projects()

    def _build_header(self) -> None:
        """Build the project workflow header."""
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, padx=12, pady=(8, 4), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="WakeLab",
            font=("Arial", 18, "bold"),
        ).grid(row=0, column=0, padx=(16, 12), pady=12, sticky="w")
        self.project_menu = ctk.CTkOptionMenu(
            header,
            variable=self.selected_project_var,
            values=["No saved projects"],
            command=self.open_selected_project,
            width=320,
        )
        self.project_menu.grid(
            row=0,
            column=1,
            padx=(0, 8),
            pady=12,
            sticky="ew",
        )
        ctk.CTkButton(
            header,
            text="Open Folder",
            width=104,
            command=self.open_selected_project_folder,
        ).grid(row=0, column=2, padx=4, pady=12)
        ctk.CTkButton(
            header,
            text="Refresh",
            width=78,
            command=self.refresh_projects,
        ).grid(row=0, column=3, padx=4, pady=12)
        ctk.CTkButton(
            header,
            text="Delete",
            width=78,
            command=self.delete_project,
        ).grid(row=0, column=4, padx=4, pady=12)
        ctk.CTkButton(
            header,
            text="New Project",
            width=108,
            command=self.new_project,
        ).grid(row=0, column=5, padx=(4, 12), pady=12)

        appearance = ctk.CTkFrame(header, fg_color="transparent")
        appearance.grid(row=0, column=6, padx=(0, 16), pady=12, sticky="e")
        ctk.CTkLabel(appearance, text="Appearance").pack(
            side="left",
            padx=(0, 8),
        )
        self.appearance_mode_selector = ctk.CTkSegmentedButton(
            appearance,
            values=["Dark", "Light"],
            command=self._set_appearance_mode,
        )
        self.appearance_mode_selector.pack(side="left")
        self.appearance_mode_selector.set("Dark")

    def _build_workflow_strip(self) -> None:
        """Build a compact workflow orientation strip."""
        workflow = ctk.CTkFrame(self, fg_color="transparent")
        workflow.grid(row=1, column=0, padx=12, pady=(0, 4), sticky="ew")
        workflow.grid_columnconfigure(0, weight=1)
        strip = ctk.CTkFrame(workflow)
        strip.grid(row=0, column=0, sticky="ew")
        strip.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        steps = [
            "Project Setup",
            "Data",
            "Training",
            "Testing",
            "Export",
        ]
        for index, step in enumerate(steps):
            ctk.CTkLabel(
                strip,
                text=f"{index + 1}. {step}",
                anchor="center",
                font=("Arial", 12, "bold" if index < 2 else "normal"),
            ).grid(
                row=0,
                column=index,
                padx=8,
                pady=8,
                sticky="ew",
            )

    def _set_appearance_mode(self, mode: str) -> None:
        """Forward appearance changes to the main application."""
        if hasattr(self.app, "set_appearance_mode"):
            self.app.set_appearance_mode(mode)
        else:
            ctk.set_appearance_mode(mode)

    def _add_section(
        self,
        parent: ctk.CTkFrame,
        *,
        row: int,
        column: int,
        title: str,
        subtitle: str,
        columnspan: int = 1,
    ) -> ctk.CTkFrame:
        """Create a titled form section.

        Args:
            parent (ctk.CTkFrame): Parent container.
            row (int): Grid row.
            column (int): Grid column.
            title (str): Section title.
            subtitle (str): Short supporting copy.
            columnspan (int): Number of columns to span.

        Returns:
            ctk.CTkFrame: Section body frame.
        """
        section = ctk.CTkFrame(parent, corner_radius=8)
        section.grid(
            row=row,
            column=column,
            columnspan=columnspan,
            padx=8,
            pady=8,
            sticky="nsew",
        )
        section.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            section,
            text=title,
            font=("Arial", 15, "bold"),
            anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(14, 2), sticky="ew")
        ctk.CTkLabel(
            section,
            text=subtitle,
            font=("Arial", 11),
            anchor="w",
            justify="left",
            wraplength=450 if columnspan == 1 else 940,
        ).grid(row=1, column=0, padx=16, pady=(0, 10), sticky="ew")
        body = ctk.CTkFrame(section, fg_color="transparent")
        body.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="ew")
        body.grid_columnconfigure(1, weight=1)
        return body

    def _add_collapsible_section(
        self,
        parent: ctk.CTkFrame,
        *,
        row: int,
        title: str,
        subtitle: str,
    ) -> ctk.CTkFrame:
        """Create the collapsed advanced settings section.

        Args:
            parent (ctk.CTkFrame): Parent container.
            row (int): Grid row.
            title (str): Section title.
            subtitle (str): Short supporting copy.

        Returns:
            ctk.CTkFrame: Hidden advanced settings body.
        """
        section = ctk.CTkFrame(parent, corner_radius=8)
        section.grid(
            row=row,
            column=0,
            columnspan=2,
            padx=8,
            pady=8,
            sticky="ew",
        )
        section.grid_columnconfigure(0, weight=1)
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.grid(row=0, column=0, padx=16, pady=(14, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text=title,
            font=("Arial", 14, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        self.advanced_toggle_button = ctk.CTkButton(
            header,
            text="Show",
            width=70,
            command=self._toggle_advanced_settings,
        )
        self.advanced_toggle_button.grid(row=0, column=1, padx=(12, 0))
        ctk.CTkLabel(
            section,
            text=subtitle,
            font=("Arial", 11),
            anchor="w",
            justify="left",
            wraplength=940,
        ).grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.advanced_body = ctk.CTkFrame(section, fg_color="transparent")
        self.advanced_body.grid_columnconfigure(1, weight=1)
        self.advanced_visible = False
        return self.advanced_body

    def _toggle_advanced_settings(self) -> None:
        """Show or hide advanced infrastructure fields."""
        if self.advanced_visible:
            self.advanced_body.grid_forget()
            self.advanced_toggle_button.configure(text="Show")
            self.advanced_visible = False
            return
        self.advanced_body.grid(
            row=2,
            column=0,
            padx=16,
            pady=(0, 16),
            sticky="ew",
        )
        self.advanced_toggle_button.configure(text="Hide")
        self.advanced_visible = True

    def _add_entry_row(
        self,
        parent: ctk.CTkFrame,
        *,
        row: int,
        label: str,
        variable: ctk.StringVar,
        browse: bool = False,
        buttons: list[tuple[str, object, int]] | None = None,
    ) -> None:
        """Add a labelled entry row with compact action buttons.

        Args:
            parent (ctk.CTkFrame): Parent form frame.
            row (int): Grid row.
            label (str): Field label.
            variable (ctk.StringVar): Entry variable.
            browse (bool): Whether to include a compact path picker button.
            buttons (list[tuple[str, object, int]] | None): Extra buttons as
                label, callback, width tuples.
        """
        ctk.CTkLabel(parent, text=label, anchor="w").grid(
            row=row,
            column=0,
            padx=(0, 12),
            pady=6,
            sticky="w",
        )
        field = ctk.CTkFrame(parent, fg_color="transparent")
        field.grid(row=row, column=1, pady=6, sticky="ew")
        field.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(field, textvariable=variable).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        next_column = 1
        if browse:
            ctk.CTkButton(
                field,
                text="...",
                width=38,
                command=lambda v=variable: self._browse_path(v),
            ).grid(row=0, column=next_column, padx=(6, 0))
            next_column += 1
        for text, command, width in buttons or []:
            ctk.CTkButton(
                field,
                text=text,
                width=width,
                command=command,
            ).grid(row=0, column=next_column, padx=(6, 0))
            next_column += 1

    def _add_option_row(
        self,
        parent: ctk.CTkFrame,
        *,
        row: int,
        label: str,
        variable: ctk.StringVar,
        values: list[str],
    ) -> None:
        """Add a labelled option row.

        Args:
            parent (ctk.CTkFrame): Parent form frame.
            row (int): Grid row.
            label (str): Field label.
            variable (ctk.StringVar): Selected value.
            values (list[str]): Option values.
        """
        ctk.CTkLabel(parent, text=label, anchor="w").grid(
            row=row,
            column=0,
            padx=(0, 12),
            pady=6,
            sticky="w",
        )
        ctk.CTkOptionMenu(
            parent,
            variable=variable,
            values=values,
            width=180,
        ).grid(row=row, column=1, pady=6, sticky="w")

    def _add_feature_bundle_panel(
        self,
        parent: ctk.CTkFrame,
        *,
        row: int,
    ) -> None:
        """Add feature bundle controls inside the training-data section."""
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=row, column=0, columnspan=2, pady=(12, 0), sticky="ew")
        panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            panel,
            text="Feature bundle",
            font=("Arial", 13, "bold"),
            anchor="w",
        ).grid(row=0, column=0, pady=(0, 8), sticky="ew")
        actions = ctk.CTkFrame(panel, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="w")
        feature_buttons = [
            ("Detect", self.detect_feature_bundle, 86),
            ("Register", self.register_existing_feature_bundle, 96),
            ("Open Folder", self.open_feature_folder, 108),
            ("Download Standard", self.download_standard_feature_bundle, 148),
        ]
        for index, (text, command, width) in enumerate(feature_buttons):
            ctk.CTkButton(
                actions,
                text=text,
                width=width,
                command=command,
            ).grid(row=0, column=index, padx=(0, 8), pady=(0, 8))
        ctk.CTkLabel(
            panel,
            text=(
                "Simple mode fills feature paths from the managed bundle when "
                "it exists. Manual paths remain available for advanced data."
            ),
            font=("Arial", 10, "italic"),
            wraplength=940,
            justify="left",
            anchor="w",
        ).grid(row=2, column=0, pady=(0, 4), sticky="ew")
        ctk.CTkLabel(
            panel,
            textvariable=self.feature_bundle_status_var,
            anchor="w",
            justify="left",
            wraplength=940,
        ).grid(row=3, column=0, sticky="ew")

    def _add_real_positives_panel(
        self,
        parent: ctk.CTkFrame,
        *,
        row: int,
    ) -> None:
        """Add real-positive recording controls."""
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=row, column=0, columnspan=2, pady=(12, 0), sticky="ew")
        panel.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            panel,
            text="Real positive recordings",
            font=("Arial", 13, "bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, pady=(0, 8), sticky="ew")
        ctk.CTkCheckBox(
            panel,
            text="Use imported real positives during training",
            variable=self.enable_real_positives_var,
        ).grid(row=1, column=0, columnspan=2, pady=4, sticky="w")
        ctk.CTkLabel(panel, text="Directory", anchor="w").grid(
            row=2,
            column=0,
            padx=(0, 12),
            pady=4,
            sticky="w",
        )
        ctk.CTkLabel(
            panel,
            textvariable=self.real_positive_dir_var,
            anchor="w",
        ).grid(
            row=2,
            column=1,
            pady=4,
            sticky="ew",
        )
        ctk.CTkLabel(panel, text="Minimum clips", anchor="w").grid(
            row=3,
            column=0,
            padx=(0, 12),
            pady=4,
            sticky="w",
        )
        ctk.CTkEntry(
            panel,
            textvariable=self.real_positive_min_count_var,
            width=120,
        ).grid(row=3, column=1, pady=4, sticky="w")
        ctk.CTkLabel(panel, text="Real training mix", anchor="w").grid(
            row=4,
            column=0,
            padx=(0, 12),
            pady=4,
            sticky="w",
        )
        ctk.CTkOptionMenu(
            panel,
            variable=self.real_positive_target_percent_var,
            values=[f"{value}%" for value in range(10, 101, 10)],
            width=120,
        ).grid(row=4, column=1, pady=4, sticky="w")
        actions = ctk.CTkFrame(panel, fg_color="transparent")
        actions.grid(row=5, column=0, columnspan=2, sticky="w")
        buttons = [
            ("Import Real Positive WAVs", self.import_real_positive_wavs, 176),
            ("Open Real Positives Folder", self.open_real_positives_folder, 188),
            ("Validate Real Positive Clips", self.validate_real_positive_clips, 198),
        ]
        for index, (text, command, width) in enumerate(buttons):
            ctk.CTkButton(
                actions,
                text=text,
                width=width,
                command=command,
            ).grid(row=0, column=index, padx=(0, 8), pady=(8, 0))

    def _build_footer(self, parent: ctk.CTkFrame, *, row: int) -> None:
        """Build the final project action area."""
        footer = ctk.CTkFrame(parent, corner_radius=8)
        footer.grid(
            row=row,
            column=0,
            columnspan=2,
            padx=8,
            pady=(8, 16),
            sticky="ew",
        )
        footer.grid_columnconfigure(1, weight=1)
        actions = ctk.CTkFrame(footer, fg_color="transparent")
        actions.grid(row=0, column=0, padx=16, pady=14, sticky="w")
        ctk.CTkButton(
            actions,
            text="Create / Save Project",
            width=160,
            command=self.create_project,
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            actions,
            text="Initialise Wake Lab Folders",
            width=190,
            command=self._initialize_folders,
        ).pack(side="left")
        ctk.CTkLabel(
            footer,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            wraplength=560,
        ).grid(row=0, column=1, padx=(10, 16), pady=14, sticky="ew")

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

    def _browse_path(self, variable: ctk.StringVar) -> None:
        initial_dir = None
        if variable is self.background_paths_var:
            initial_dir = wake_lab_home.get_background_audio_dir()
        elif variable is self.rir_paths_var:
            initial_dir = wake_lab_home.get_rir_dir()
        elif variable is self.negative_feature_paths_var:
            initial_dir = wake_lab_home.get_negative_features_dir()
        elif variable is self.validation_path_var:
            initial_dir = wake_lab_home.get_false_positive_validation_dir()
        elif variable is self.piper_model_path_var:
            initial_dir = wake_lab_home.get_downloads_dir()
        elif variable is self.workspace_root_var:
            initial_dir = wake_lab_home.get_projects_root()
        elif variable is self.oww_path_var:
            initial_dir = wake_lab_home.detect_openwakeword_repo()
        elif variable is self.orac_path_var:
            initial_dir = wake_lab_home.detect_orac_repo()
        elif variable is self.piper_path_var:
            initial_dir = wake_lab_home.detect_piper_sample_generator_path()

        if initial_dir and not initial_dir.exists():
            initial_dir = wake_lab_home.get_wake_lab_home()

        initial_dir_str = str(initial_dir) if initial_dir else None

        if variable is self.negative_feature_paths_var:
            selected_files = filedialog.askopenfilenames(
                initialdir=initial_dir_str,
                filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")]
            )
            selected = ",".join(selected_files)
        elif variable is self.validation_path_var:
            selected = filedialog.askopenfilename(
                initialdir=initial_dir_str,
                filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")]
            )
        elif variable is self.piper_model_path_var:
            selected = filedialog.askopenfilename(
                initialdir=initial_dir_str,
                filetypes=[
                    ("Piper models", "*.onnx"),
                    ("Piper models", "*.pt"),
                    ("All files", "*.*"),
                ],
            )
        else:
            selected = filedialog.askdirectory(initialdir=initial_dir_str)

        if selected:
            variable.set(selected)
            if (
                variable is self.negative_feature_paths_var
                or variable is self.validation_path_var
            ):
                self._sync_current_project_feature_paths()
            elif variable is self.piper_model_path_var:
                self._sync_current_project_piper_model_path()

    def refresh_feature_bundle(
        self,
        *,
        auto_apply: bool = False,
        force: bool = False,
    ) -> feature_bundle.FeatureBundleCheck:
        """Refresh the managed standard feature bundle state.

        Args:
            auto_apply (bool): Populate blank project fields from the managed
                bundle when available.
            force (bool): Replace the current project feature paths.

        Returns:
            feature_bundle.FeatureBundleCheck: Bundle validation state.
        """
        bundle = feature_bundle.detect_standard_feature_bundle()
        if bundle.is_ready and (force or self._feature_fields_are_blank()):
            self._apply_feature_bundle_paths(bundle.paths)
            self.feature_bundle_status_var.set(
                "Simple mode filled the project from the managed feature bundle."
            )
        elif bundle.is_ready:
            self.feature_bundle_status_var.set(
                "Standard feature bundle is available, but the project is using manually configured feature paths."
            )
        else:
            self.feature_bundle_status_var.set(bundle.status_message)
        return bundle

    def detect_feature_bundle(self) -> None:
        """Detect the managed feature bundle and populate the form."""
        bundle = self.refresh_feature_bundle(force=True)
        if bundle.is_ready:
            self.feature_bundle_status_var.set(
                "Managed feature bundle detected and loaded into the project."
            )

    def register_existing_feature_bundle(self) -> None:
        """Register an existing feature bundle with the managed folders."""
        negative_source = filedialog.askopenfilename(
            title="Select the negative/background feature .npy file",
            initialdir=str(wake_lab_home.get_negative_features_dir()),
            filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
        )
        if not negative_source:
            return
        validation_source = filedialog.askopenfilename(
            title="Select the false-positive validation .npy file",
            initialdir=str(wake_lab_home.get_false_positive_validation_dir()),
            filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
        )
        if not validation_source:
            return
        try:
            bundle = feature_bundle.register_existing_feature_bundle(
                Path(negative_source),
                Path(validation_source),
            )
        except Exception as exc:
            self.feature_bundle_status_var.set(str(exc))
            return
        self._apply_feature_bundle_paths(bundle.paths)
        self.feature_bundle_status_var.set(
            "Existing feature bundle registered in the managed WakeLab folders."
        )

    def open_feature_folder(self) -> None:
        """Open the managed feature folder in the file browser."""
        try:
            feature_bundle.open_feature_bundle_folder()
        except Exception as exc:
            self.feature_bundle_status_var.set(str(exc))
            return
        self.feature_bundle_status_var.set(
            str(wake_lab_home.get_features_dir())
        )

    def download_standard_feature_bundle(self) -> None:
        """Download the standard openWakeWord feature bundle."""
        confirmed = messagebox.askyesno(
            "Download Standard Feature Bundle",
            (
                "WakeLab will download the standard openWakeWord feature "
                "bundle into the managed WakeLab folders.\n\n"
                f"The negative/background feature file is large: "
                f"{feature_bundle.STANDARD_FEATURE_BUNDLE_SIZE_NOTE}.\n"
                "The validation file is much smaller.\n\n"
                "This may take a long time and use a lot of disk space.\n\n"
                "Continue?"
            ),
        )
        if not confirmed:
            self.feature_bundle_status_var.set("Download cancelled.")
            return

        self.feature_bundle_status_var.set("Downloading standard feature bundle...")
        threading.Thread(
            target=self._download_standard_feature_bundle_worker,
            daemon=True,
        ).start()

    def _download_standard_feature_bundle_worker(self) -> None:
        """Download the bundle on a background thread."""

        def progress_callback(filename: str, downloaded: int, total: int) -> None:
            if total > 0:
                percent = (downloaded / total) * 100
                status = (
                    f"Downloading {filename}: {percent:.1f}% "
                    f"({downloaded / (1024**3):.1f}/{total / (1024**3):.1f} GB)"
                )
            else:
                status = (
                    f"Downloading {filename}: "
                    f"{downloaded / (1024**3):.1f} GB downloaded"
                )
            self.after(0, self.feature_bundle_status_var.set, status)

        try:
            bundle = feature_bundle.download_standard_feature_bundle(
                progress_callback=progress_callback
            )
        except Exception as exc:
            self.after(0, self.feature_bundle_status_var.set, str(exc))
            return
        self.after(0, self._apply_downloaded_feature_bundle, bundle)

    def _apply_downloaded_feature_bundle(
        self,
        bundle: feature_bundle.FeatureBundleCheck,
    ) -> None:
        """Update the UI after a successful feature bundle download."""
        self._apply_feature_bundle_paths(bundle.paths)
        self.feature_bundle_status_var.set(
            "Standard feature bundle downloaded and registered."
        )

    def _apply_feature_bundle_paths(
        self,
        paths: feature_bundle.FeatureBundlePaths,
    ) -> None:
        """Populate the form from managed feature bundle paths."""
        self.negative_feature_paths_var.set(str(paths.negative))
        self.validation_path_var.set(str(paths.validation))
        self._sync_current_project_feature_paths()

    def _feature_fields_are_blank(self) -> bool:
        """Return whether the feature fields are empty."""
        return not self.negative_feature_paths_var.get().strip() and not self.validation_path_var.get().strip()

    def _sync_current_project_feature_paths(self) -> None:
        """Copy feature-path fields into the active project, if any."""
        project = self.app.get_project()
        if project is None:
            return
        project.negative_feature_data_files = _split_feature_files(
            self.negative_feature_paths_var.get()
        )
        project.false_positive_validation_data_path = _path_from_entry(
            self.validation_path_var.get()
        )

    def _sync_current_project_piper_model_path(self) -> None:
        """Copy the Piper model field into the active project, if any."""
        project = self.app.get_project()
        if project is None:
            return
        project.piper_voice_model_path = _path_from_entry(
            self.piper_model_path_var.get()
        )

    def import_real_positive_wavs(self) -> None:
        """Import selected real positive WAV files into the project."""
        project = self.app.get_project()
        if project is None:
            self.status_var.set("Create or open a project before importing real positives.")
            return
        try:
            self._sync_real_positive_settings_from_form(project)
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        selected_files = filedialog.askopenfilenames(
            title="Import real positive wake-word WAVs",
            initialdir=str(project.real_positive_clips_dir),
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not selected_files:
            return
        try:
            imported = real_positives.import_real_positive_wavs(
                project,
                [Path(path) for path in selected_files],
            )
            save_project(project)
        except Exception as exc:
            self.status_var.set(f"Real positive import failed: {exc}")
            return
        self.real_positive_dir_var.set(str(project.real_positive_clips_dir))
        self.status_var.set(f"Imported {len(imported)} real positive WAV(s).")

    def open_real_positives_folder(self) -> None:
        """Open the project-local real positives folder."""
        project = self.app.get_project()
        if project is None:
            self.status_var.set("Create or open a project first.")
            return
        try:
            self._sync_real_positive_settings_from_form(project)
            project.real_positive_clips_dir.mkdir(parents=True, exist_ok=True)
            _open_folder(project.real_positive_clips_dir)
            save_project(project)
        except Exception as exc:
            self.status_var.set(f"Unable to open real positives folder: {exc}")
            return
        self.real_positive_dir_var.set(str(project.real_positive_clips_dir))
        self.status_var.set(str(project.real_positive_clips_dir))

    def validate_real_positive_clips(self) -> None:
        """Validate project-local real positive WAV clips."""
        project = self.app.get_project()
        if project is None:
            self.status_var.set("Create or open a project first.")
            return
        try:
            self._sync_real_positive_settings_from_form(project)
            summary = real_positives.real_positive_summary(project)
            save_project(project)
        except Exception as exc:
            self.status_var.set(f"Real positive validation failed: {exc}")
            return
        self.real_positive_dir_var.set(str(project.real_positive_clips_dir))
        self.status_var.set(summary)

    def _sync_real_positive_settings_from_form(
        self,
        project: WakeWordProject,
    ) -> None:
        """Copy real-positive fields into the active project."""
        try:
            minimum = int(self.real_positive_min_count_var.get())
            target_percent = _parse_percent(
                self.real_positive_target_percent_var.get()
            )
        except ValueError as exc:
            raise ValueError(
                "Real positive minimum and training mix values must be "
                "whole numbers."
            ) from exc
        if target_percent < 10 or target_percent > 100:
            raise ValueError(
                "Real positive training mix must be between 10% and 100%."
            )
        project.enable_real_positives = bool(self.enable_real_positives_var.get())
        project.real_positive_clips_dir = project.real_positives_dir
        project.real_positive_min_count = minimum
        project.real_positive_target_percent = target_percent

    def refresh_projects(self) -> None:
        """Refresh saved project choices from the managed projects root."""
        projects = discover_projects()
        self.available_project_paths = {
            _project_label(project): project.workspace_dir / "project.json"
            for project in projects
        }
        values = list(self.available_project_paths) or ["No saved projects"]
        self.project_menu.configure(values=values)
        current_project = getattr(self.app, "get_project", lambda: None)()
        current_label = (
            _project_label(current_project)
            if current_project is not None
            else ""
        )
        selected = (
            current_label
            if current_label in self.available_project_paths
            else self.selected_project_var.get()
            if self.selected_project_var.get() in self.available_project_paths
            else values[0]
        )
        self.selected_project_var.set(selected)

    def new_project(self) -> None:
        """Reset the project form to defaults for a new project."""
        self._apply_form_values(_new_project_form_values())
        if hasattr(self.app, "clear_project"):
            self.app.clear_project()
        self.refresh_feature_bundle(auto_apply=True)
        self.status_var.set("New project form ready.")

    def open_selected_project(self, selected: str | None = None) -> None:
        """Open the selected saved project.

        Args:
            selected (str | None): Selected project label from the dropdown.
        """
        if selected is not None:
            self.selected_project_var.set(selected)
        selected = self.selected_project_var.get()
        project_path = self.available_project_paths.get(selected)
        if project_path is None:
            self.status_var.set("No saved project selected.")
            return
        try:
            self.open_project(project_path)
        except Exception as exc:
            self.status_var.set(f"Unable to load project: {exc}")
            self.refresh_projects()

    def open_project(self, project_path: Path) -> WakeWordProject:
        """Load an existing project and populate the form state.

        Args:
            project_path (Path): Path to the saved project JSON file.

        Returns:
            WakeWordProject: Loaded project.
        """
        project = load_project(project_path)
        self.app.set_project(project)
        self.apply_project(project)
        self.selected_project_var.set(_project_label(project))
        self.status_var.set(f"Project loaded: {project.workspace_dir}.")
        return project

    def open_selected_project_folder(self) -> None:
        """Open the selected or current project workspace in the file browser."""
        project = getattr(self.app, "get_project", lambda: None)()
        if project is not None:
            _open_folder(project.workspace_dir)
            self.status_var.set(
                f"Opened project folder: {project.workspace_dir}."
            )
            return
        selected = self.selected_project_var.get()
        project_path = self.available_project_paths.get(selected)
        if project_path is None:
            self.status_var.set("No saved project selected.")
            return
        _open_folder(project_path.parent)
        self.status_var.set(f"Opened project folder: {project_path.parent}.")

    def delete_project(self) -> None:
        """Delete the active or selected saved project workspace."""
        if _training_job_is_running(self.app):
            self.status_var.set(
                "Stop the current training job before deleting a project."
            )
            return

        project = self.app.get_project()
        if project is None:
            selected = self.selected_project_var.get()
            project_path = self.available_project_paths.get(selected)
            if project_path is None:
                self.status_var.set("No saved project selected.")
                return
            project = load_project(project_path)

        confirmed = messagebox.askyesno(
            "Delete Project",
            (
                "Delete this WakeLab project workspace?\n\n"
                f"Project: {project.model_name}\n"
                f"Workspace: {project.workspace_dir}\n\n"
                "This removes generated clips, logs, models, exports, and "
                "project settings for this project."
            ),
        )
        if not confirmed:
            self.status_var.set("Project deletion cancelled.")
            return

        was_current_project = _same_workspace(
            project,
            self.app.get_project(),
        )
        try:
            deleted = delete_project_workspace(project)
        except Exception as exc:
            self.status_var.set(f"Unable to delete project: {exc}")
            return

        self.refresh_projects()
        if was_current_project and hasattr(self.app, "clear_project"):
            self.app.clear_project()
            self._apply_form_values(_new_project_form_values())
            self.refresh_feature_bundle(auto_apply=True)

        if deleted:
            self.status_var.set(f"Project deleted: {project.workspace_dir}.")
        else:
            self.status_var.set(
                f"Project workspace was already missing: {project.workspace_dir}."
            )

    def apply_project(self, project: WakeWordProject) -> None:
        """Populate editable fields from a loaded project.

        Args:
            project (WakeWordProject): Loaded project data.
        """
        self._apply_form_values(project_form_values(project))
        self.refresh_feature_bundle(auto_apply=True)

    def _apply_form_values(self, values: dict[str, str]) -> None:
        """Populate editable fields from form values.

        Args:
            values (dict[str, str]): Form field values.
        """
        self.wake_phrase_var.set(values["wake_phrase"])
        self.model_name_var.set(values["model_name"])
        self.workspace_root_var.set(values["workspace_root"])
        self.oww_path_var.set(values["openwakeword_repo"])
        self.orac_path_var.set(values["orac_repo"])
        self.piper_path_var.set(values["piper_sample_generator_path"])
        self.piper_model_path_var.set(values["piper_voice_model_path"])
        self.training_pronunciation_phrase_var.set(
            values["training_pronunciation_phrase"]
        )
        self.training_phrase_parts_var.set(values["training_phrase_parts"])
        self.inter_part_silence_min_var.set(
            values["inter_part_silence_min_ms"]
        )
        self.inter_part_silence_max_var.set(
            values["inter_part_silence_max_ms"]
        )
        self.enable_real_positives_var.set(
            values["enable_real_positives"] == "true"
        )
        self.real_positive_dir_var.set(values["real_positive_clips_dir"])
        self.real_positive_min_count_var.set(
            values["real_positive_min_count"]
        )
        self.real_positive_target_percent_var.set(
            values["real_positive_target_percent"]
        )
        self.background_paths_var.set(values["background_paths"])
        self.rir_paths_var.set(values["rir_paths"])
        self.negative_feature_paths_var.set(
            values["negative_feature_data_files"]
        )
        self.validation_path_var.set(
            values["false_positive_validation_data_path"]
        )
        self.negatives_var.set(values["custom_negative_phrases"])
        self.profile_var.set(values["profile"])

    def test_piper_voice_model(self) -> None:
        """Synthesise and play a short sample using the selected Piper model."""
        model_path = Path(self.piper_model_path_var.get().strip())
        generator_path = Path(self.piper_path_var.get().strip())
        if not self.piper_model_path_var.get().strip():
            self.status_var.set("Select a Piper voice or generator model first.")
            return

        self.status_var.set("Testing Piper voice model...")
        threading.Thread(
            target=self._test_piper_voice_model_worker,
            args=(model_path, generator_path),
            daemon=True,
        ).start()

    def _test_piper_voice_model_worker(
        self,
        model_path: Path,
        generator_path: Path,
    ) -> None:
        """Run voice synthesis and playback on a background thread."""
        try:
            piper_voice_test.test_piper_voice(model_path, generator_path)
        except Exception as exc:
            self.after(
                0,
                self.status_var.set,
                f"Piper voice test failed: {exc}",
            )
            return
        self.after(
            0,
            self.status_var.set,
            "Piper voice test played successfully.",
        )

    def create_project(self) -> None:
        """Create or update the current wake-word project."""
        self.refresh_feature_bundle(auto_apply=True)
        phrase = self.wake_phrase_var.get().strip()
        model_name = self.model_name_var.get().strip()
        phrase_result = validate_phrase(phrase)
        model_result = validate_model_name(model_name)
        if phrase_result.is_failure or model_result.is_failure:
            self.status_var.set(
                f"{phrase_result.message} {model_result.message}"
            )
            return
        try:
            silence_min_ms = int(self.inter_part_silence_min_var.get())
            silence_max_ms = int(self.inter_part_silence_max_var.get())
            real_positive_min_count = int(
                self.real_positive_min_count_var.get()
            )
            real_positive_target_percent = _parse_percent(
                self.real_positive_target_percent_var.get()
            )
        except ValueError:
            self.status_var.set(
                "Inter-part silence and real positive values must be whole "
                "numbers."
            )
            return
        if (
            real_positive_target_percent < 10
            or real_positive_target_percent > 100
        ):
            self.status_var.set(
                "Real positive training mix must be between 10% and 100%."
            )
            return
        training_phrase_parts = _split_phrase_parts(
            self.training_phrase_parts_var.get()
        )
        if any(not part for part in training_phrase_parts):
            self.status_var.set("Training phrase parts must not be blank.")
            return

        workspace_dir = project_dir_for_model(
            model_name,
            Path(self.workspace_root_var.get()),
        )
        real_positive_dir = workspace_dir / "real_positives"
        project = WakeWordProject(
            wake_phrase=phrase,
            model_name=model_name,
            workspace_dir=workspace_dir,
            openwakeword_repo=Path(self.oww_path_var.get()),
            orac_repo=Path(self.orac_path_var.get()),
            piper_sample_generator_path=Path(self.piper_path_var.get()),
            piper_voice_model_path=_path_from_entry(
                self.piper_model_path_var.get()
            ),
            training_pronunciation_phrase=(
                self.training_pronunciation_phrase_var.get().strip()
            ),
            training_phrase_parts=training_phrase_parts,
            inter_part_silence_min_ms=silence_min_ms,
            inter_part_silence_max_ms=silence_max_ms,
            enable_real_positives=bool(
                self.enable_real_positives_var.get()
            ),
            real_positive_clips_dir=real_positive_dir,
            real_positive_min_count=real_positive_min_count,
            real_positive_target_percent=real_positive_target_percent,
            background_paths=_split_paths(self.background_paths_var.get()),
            rir_paths=_split_paths(self.rir_paths_var.get()),
            negative_feature_data_files=_split_feature_files(
                self.negative_feature_paths_var.get()
            ),
            false_positive_validation_data_path=_path_from_entry(
                self.validation_path_var.get()
            ),
            custom_negative_phrases=_split_values(self.negatives_var.get()),
            profile=self.profile_var.get(),
        )
        positive_generation_result = validate_positive_generation_settings(
            project
        )
        if positive_generation_result.is_failure:
            self.status_var.set(positive_generation_result.message)
            return
        text_result = validate_training_text_fields(project)
        if text_result.is_failure:
            self.status_var.set(text_result.message)
            return
        create_project_workspace(project)
        write_training_config(project)
        self.app.set_project(project)
        warning = (
            f" Warning: {phrase_result.message}"
            if phrase_result.status == "warn"
            else ""
        )
        self.status_var.set(f"Project saved: {workspace_dir}.{warning}")
        self.refresh_projects()


def _split_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_phrase_parts(value: str) -> list[str]:
    """Split the phrase-parts field, preserving blank enabled parts."""
    if not value.strip():
        return []
    return [item.strip() for item in value.split("|")]


def _split_feature_files(value: str) -> dict[str, Path]:
    paths = _split_paths(value)
    return {path.stem: path for path in paths}


def _same_workspace(
    left: WakeWordProject | None,
    right: WakeWordProject | None,
) -> bool:
    """Return whether two projects refer to the same workspace."""
    if left is None or right is None:
        return False
    return left.workspace_dir.expanduser().resolve(
        strict=False
    ) == right.workspace_dir.expanduser().resolve(strict=False)


def _training_job_is_running(app: object) -> bool:
    """Return whether the app has a running training job."""
    training_tab = getattr(app, "training_tab", None)
    runner = getattr(training_tab, "runner", None)
    return bool(getattr(runner, "is_running", False))


def _validate_project_child_path(
    project: WakeWordProject,
    path: Path,
) -> None:
    """Validate that a path is inside the project workspace."""
    _validate_project_child_path_for_workspace(project.workspace_dir, path)


def _validate_project_child_path_for_workspace(
    workspace_dir: Path,
    path: Path,
) -> None:
    """Validate that a path is inside a workspace directory."""
    workspace = workspace_dir.expanduser().resolve(strict=False)
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(
            "Real positive clips must live under the project workspace."
        ) from exc


def _open_folder(folder: Path) -> None:
    """Open a folder in the platform file browser."""
    if os.name == "nt":
        os.startfile(folder)  # type: ignore[attr-defined]
        return
    if platform.system() == "Darwin":
        subprocess.Popen(["open", str(folder)])
        return
    subprocess.Popen(["xdg-open", str(folder)])


def project_form_values(project: WakeWordProject) -> dict[str, str]:
    """Return string values used to populate the project form.

    Args:
        project (WakeWordProject): Project to display.

    Returns:
        dict[str, str]: Form field values.
    """
    return {
        "wake_phrase": project.wake_phrase,
        "model_name": project.model_name,
        "workspace_root": str(project.workspace_dir.parent),
        "openwakeword_repo": str(project.openwakeword_repo),
        "orac_repo": str(project.orac_repo),
        "piper_sample_generator_path": str(
            project.piper_sample_generator_path
        ),
        "piper_voice_model_path": _format_path_for_entry(
            project.piper_voice_model_path
        ),
        "training_pronunciation_phrase": (
            project.training_pronunciation_phrase
        ),
        "training_phrase_parts": " | ".join(
            project.training_phrase_parts or []
        ),
        "inter_part_silence_min_ms": str(project.inter_part_silence_min_ms),
        "inter_part_silence_max_ms": str(project.inter_part_silence_max_ms),
        "enable_real_positives": (
            "true" if project.enable_real_positives else "false"
        ),
        "real_positive_clips_dir": str(project.real_positive_clips_dir),
        "real_positive_min_count": str(project.real_positive_min_count),
        "real_positive_target_percent": (
            f"{project.real_positive_target_percent}%"
        ),
        "background_paths": ",".join(
            str(path) for path in project.background_paths
        ),
        "rir_paths": ",".join(str(path) for path in project.rir_paths),
        "negative_feature_data_files": ",".join(
            str(path) for path in project.negative_feature_data_files.values()
        ),
        "false_positive_validation_data_path": _format_path_for_entry(
            project.false_positive_validation_data_path
        ),
        "custom_negative_phrases": ",".join(
            project.custom_negative_phrases
        ),
        "profile": project.profile,
    }


def _default_project_form_values() -> dict[str, str]:
    """Return the default form values used when the tab starts."""
    return {
        "wake_phrase": "Hey Orac",
        "model_name": "hey_orac",
        "workspace_root": str(DEFAULT_WORKSPACE_ROOT),
        "openwakeword_repo": str(wake_lab_home.get_openwakeword_repo_dir()),
        "orac_repo": str(wake_lab_home.get_orac_repo_dir()),
        "piper_sample_generator_path": str(
            wake_lab_home.get_piper_sample_generator_dir()
        ),
        "piper_voice_model_path": "",
        "training_pronunciation_phrase": "",
        "training_phrase_parts": "",
        "inter_part_silence_min_ms": "80",
        "inter_part_silence_max_ms": "250",
        "enable_real_positives": "true",
        "real_positive_clips_dir": "",
        "real_positive_min_count": "20",
        "real_positive_target_percent": "30%",
        "background_paths": str(wake_lab_home.get_background_audio_dir()),
        "rir_paths": str(wake_lab_home.get_rir_dir()),
        "negative_feature_data_files": "",
        "false_positive_validation_data_path": "",
        "custom_negative_phrases": "hey oracle, oracle, orac",
        "profile": "quick",
    }


def _new_project_form_values() -> dict[str, str]:
    """Return blank project-specific values for a new project."""
    values = _default_project_form_values()
    values.update(
        {
            "wake_phrase": "",
            "model_name": "",
            "real_positive_clips_dir": "",
            "custom_negative_phrases": "",
        }
    )
    return values


def _project_label(project: WakeWordProject) -> str:
    return f"{project.model_name} ({project.wake_phrase})"


def _path_from_entry(value: str) -> Path:
    """Convert a text field value into a path, preserving blank input.

    Args:
        value (str): Entry text.

    Returns:
        Path: Parsed path, or an empty path for blank input.
    """
    cleaned = value.strip()
    return Path(cleaned) if cleaned else Path("")


def _format_path_for_entry(path: Path) -> str:
    """Format a path for display in a text field.

    Args:
        path (Path): Path to display.

    Returns:
        str: Empty string when the path is unset.
    """
    value = str(path).strip()
    return "" if value in {"", "."} else value


def _parse_percent(value: str) -> int:
    """Parse a percentage value from a form control."""
    return int(value.strip().removesuffix("%"))
