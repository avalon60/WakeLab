"""Project setup tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Builds the wake phrase and workspace setup UI.

from __future__ import annotations

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
from orac_wake_lab.services import wake_lab_home
from orac_wake_lab.services.project_store import create_project_workspace
from orac_wake_lab.services.project_store import discover_projects
from orac_wake_lab.services.project_store import load_project
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
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Saved projects").grid(
            row=0,
            column=0,
            padx=12,
            pady=6,
            sticky="w",
        )
        self.project_menu = ctk.CTkOptionMenu(
            self,
            variable=self.selected_project_var,
            values=["No saved projects"],
        )
        self.project_menu.grid(row=0, column=1, padx=12, pady=6, sticky="ew")
        project_buttons = ctk.CTkFrame(self, fg_color="transparent")
        project_buttons.grid(row=0, column=2, padx=12, pady=6, sticky="w")
        ctk.CTkButton(
            project_buttons,
            text="New Project",
            width=110,
            command=self.new_project,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            project_buttons,
            text="Open",
            width=80,
            command=self.open_selected_project,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            project_buttons,
            text="Refresh",
            width=80,
            command=self.refresh_projects,
        ).pack(side="left")

        rows = [
            ("Wake phrase", self.wake_phrase_var),
            ("Model name", self.model_name_var),
            ("Workspace root", self.workspace_root_var),
            ("Local openWakeWord checkout", self.oww_path_var),
            ("Local Orac checkout", self.orac_path_var),
            ("Local Piper sample generator checkout", self.piper_path_var),
            ("Piper voice / generator model", self.piper_model_path_var),
            ("Background audio dirs", self.background_paths_var),
            ("RIR dirs", self.rir_paths_var),
            ("Negative feature .npy files", self.negative_feature_paths_var),
            ("False-positive validation .npy", self.validation_path_var),
            ("Negative phrases", self.negatives_var),
        ]
        
        current_row = 1
        for label, variable in rows:
            ctk.CTkLabel(self, text=label).grid(
                row=current_row,
                column=0,
                padx=12,
                pady=6,
                sticky="w",
            )
            ctk.CTkEntry(self, textvariable=variable).grid(
                row=current_row,
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
                ).grid(row=current_row, column=2, padx=12, pady=6)
            if self._field_uses_path_browser(variable, label):
                ctk.CTkButton(
                    self,
                    text="Browse",
                    width=80,
                    command=lambda v=variable, l=label: self._browse_path(v, l),
                ).grid(row=current_row, column=2, padx=12, pady=6)
            if variable is self.piper_model_path_var:
                ctk.CTkButton(
                    self,
                    text="Test",
                    width=80,
                    command=self.test_piper_voice_model,
                ).grid(row=current_row, column=3, padx=12, pady=6)

            if label == "Local Piper sample generator checkout":
                current_row += 1
                ctk.CTkLabel(
                    self,
                    text=(
                        "Used by openWakeWord to generate synthetic spoken examples "
                        "of the wake phrase."
                    ),
                    font=("Arial", 10, "italic"),
                ).grid(row=current_row, column=1, sticky="w", padx=12, pady=(0, 6))
            elif label == "Piper voice / generator model":
                current_row += 1
                ctk.CTkLabel(
                    self,
                    text=(
                        "Selecting a model updates the active project form. "
                        "Use Create / Save Project at the bottom to persist it."
                    ),
                    font=("Arial", 10, "italic"),
                ).grid(row=current_row, column=1, sticky="w", padx=12, pady=(0, 6))
            
            current_row += 1

        feature_bundle_frame = ctk.CTkFrame(self, fg_color="transparent")
        feature_bundle_frame.grid(
            row=current_row,
            column=0,
            columnspan=3,
            padx=12,
            pady=(10, 0),
            sticky="ew",
        )
        feature_bundle_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(feature_bundle_frame, text="Feature bundle").grid(
            row=0,
            column=0,
            padx=(0, 12),
            pady=(0, 6),
            sticky="w",
        )
        feature_actions = ctk.CTkFrame(feature_bundle_frame, fg_color="transparent")
        feature_actions.grid(row=0, column=1, pady=(0, 6), sticky="w")
        ctk.CTkButton(
            feature_actions,
            text="Detect Feature Bundle",
            command=self.detect_feature_bundle,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            feature_actions,
            text="Register Existing Feature Bundle",
            command=self.register_existing_feature_bundle,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            feature_actions,
            text="Open Feature Folder",
            command=self.open_feature_folder,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            feature_actions,
            text="Download Standard Feature Bundle",
            command=self.download_standard_feature_bundle,
        ).pack(side="left")
        ctk.CTkLabel(
            feature_bundle_frame,
            text=(
                "Simple mode will fill these fields from the managed feature "
                "bundle when it exists. The Browse buttons below remain "
                "available for Advanced mode overrides."
            ),
            font=("Arial", 10, "italic"),
            wraplength=820,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, padx=(0, 12), pady=(0, 4), sticky="w")
        ctk.CTkLabel(
            feature_bundle_frame,
            textvariable=self.feature_bundle_status_var,
            anchor="w",
            justify="left",
            wraplength=820,
        ).grid(row=2, column=0, columnspan=2, padx=(0, 12), pady=(0, 6), sticky="w")

        current_row += 1

        self.wake_phrase_var.trace_add("write", self._phrase_changed)

        ctk.CTkLabel(self, text="Profile").grid(
            row=current_row,
            column=0,
            padx=12,
            pady=6,
            sticky="w",
        )
        ctk.CTkOptionMenu(
            self,
            variable=self.profile_var,
            values=["quick", "balanced", "manual"],
        ).grid(row=current_row, column=1, padx=12, pady=6, sticky="w")

        current_row += 1
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=current_row, column=1, padx=12, pady=12, sticky="w")

        ctk.CTkButton(
            button_frame,
            text="Create / Save Project",
            command=self.create_project,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            button_frame,
            text="Initialise Wake Lab Folders",
            command=self._initialize_folders,
        ).pack(side="left")

        current_row += 1
        ctk.CTkLabel(self, textvariable=self.status_var).grid(
            row=current_row,
            column=0,
            columnspan=3,
            padx=12,
            pady=8,
            sticky="w",
        )
        self.refresh_projects()

    def _field_uses_path_browser(
        self,
        variable: ctk.StringVar,
        label: str,
    ) -> bool:
        """Return whether a form field should show a Browse button.

        Args:
            variable (ctk.StringVar): Field variable.
            label (str): Display label for the field.

        Returns:
            bool: Whether the field accepts a path selected by file dialog.
        """
        return (
            variable is self.oww_path_var
            or variable is self.orac_path_var
            or variable is self.piper_path_var
            or variable is self.piper_model_path_var
            or "dirs" in label.lower()
            or ".npy" in label.lower()
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
        elif variable is self.piper_model_path_var:
            initial_dir = wake_lab_home.get_downloads_dir()
        elif "openwakeword" in label.lower():
            initial_dir = wake_lab_home.detect_openwakeword_repo()
        elif "orac" in label.lower():
            initial_dir = wake_lab_home.detect_orac_repo()
        elif "piper" in label.lower():
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

    def refresh_projects(self) -> None:
        """Refresh saved project choices from the managed projects root."""
        projects = discover_projects()
        self.available_project_paths = {
            _project_label(project): project.workspace_dir / "project.json"
            for project in projects
        }
        values = list(self.available_project_paths) or ["No saved projects"]
        self.project_menu.configure(values=values)
        self.selected_project_var.set(values[0])

    def new_project(self) -> None:
        """Reset the project form to defaults for a new project."""
        self._apply_form_values(_new_project_form_values())
        if hasattr(self.app, "clear_project"):
            self.app.clear_project()
        self.refresh_feature_bundle(auto_apply=True)
        self.status_var.set("New project form ready.")

    def open_selected_project(self) -> None:
        """Open the selected saved project."""
        selected = self.selected_project_var.get()
        project_path = self.available_project_paths.get(selected)
        if project_path is None:
            self.status_var.set("No saved project selected.")
            return
        self.open_project(project_path)

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
        self.status_var.set(f"Project loaded: {project.workspace_dir}.")
        return project

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

        workspace_dir = project_dir_for_model(
            model_name,
            Path(self.workspace_root_var.get()),
        )
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


def _split_feature_files(value: str) -> dict[str, Path]:
    paths = _split_paths(value)
    return {path.stem: path for path in paths}


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
