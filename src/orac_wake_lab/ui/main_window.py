"""Main CustomTkinter window for WakeLab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Composes the WakeLab desktop UI.

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import customtkinter as ctk

from orac_wake_lab import APP_NAME
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.ui.checks_tab import ChecksTab
from orac_wake_lab.ui.export_tab import ExportTab
from orac_wake_lab.ui.project_tab import ProjectTab
from orac_wake_lab.ui.test_tab import TestTab
from orac_wake_lab.ui.training_tab import TrainingTab


THEMES_DIR = Path(__file__).resolve().parents[1] / "themes"
DEFAULT_THEME_NAME = "NightTrain"
DEFAULT_THEME_PATH = THEMES_DIR / f"{DEFAULT_THEME_NAME}.json"
THEME_SELECTION_PATH = THEMES_DIR / "set_theme.txt"
THEME_PATH = DEFAULT_THEME_PATH
DEFAULT_APPEARANCE_MODE = "Dark"
APPEARANCE_MODES = ("Dark", "Light")
WORKFLOW_STEPS = (
    ("Project Setup", "Project"),
    ("Training Data", "Project"),
    ("Training", "Train"),
    ("Testing", "Test"),
    ("Export", "Export"),
)


class ThemeSettings(NamedTuple):
    """Resolved theme file and appearance mode."""

    theme_path: Path
    appearance_mode: str


class OracWakeLabApp(ctk.CTk):
    """Main WakeLab application window."""

    def __init__(self) -> None:
        """Create the main application window."""
        theme_settings = _resolve_theme_settings()
        ctk.set_default_color_theme(str(theme_settings.theme_path))
        ctk.set_appearance_mode(theme_settings.appearance_mode)
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1100x760")
        self.current_project: WakeWordProject | None = None
        self._workflow_buttons: dict[str, ctk.CTkButton] = {}
        self._checks_button: ctk.CTkButton | None = None
        self._active_workflow_step = "Project Setup"
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(
            row=0,
            column=0,
            padx=12,
            pady=(8, 0),
            sticky="ew",
        )
        header.grid_columnconfigure(0, weight=1)

        self.project_label = ctk.CTkLabel(
            header,
            text="No project loaded.",
            anchor="w",
        )
        self.project_label.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        self._build_workflow_stepper()

        self.tabs = ctk.CTkTabview(
            self,
            fg_color="transparent",
            border_width=0,
        )
        self.tabs.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.tabs.configure(command=self._sync_workflow_from_selected_tab)
        for name in ["Project", "Checks", "Train", "Test", "Export"]:
            self.tabs.add(name)
        tab_selector = getattr(self.tabs, "_segmented_button", None)
        if tab_selector is not None:
            tab_selector.grid_forget()
        self.project_tab = ProjectTab(self.tabs.tab("Project"), self)
        self.checks_tab = ChecksTab(self.tabs.tab("Checks"), self)
        self.training_tab = TrainingTab(self.tabs.tab("Train"), self)
        self.test_tab = TestTab(self.tabs.tab("Test"), self)
        self.export_tab = ExportTab(self.tabs.tab("Export"), self)
        self.project_tab.pack(fill="both", expand=True)
        self.checks_tab.pack(fill="both", expand=True)
        self.training_tab.pack(fill="both", expand=True)
        self.test_tab.pack(fill="both", expand=True)
        self.export_tab.pack(fill="both", expand=True)
        self._update_workflow_stepper("Project Setup")

    def _build_workflow_stepper(self) -> None:
        """Build the global workflow navigation stepper."""
        workflow = ctk.CTkFrame(self, fg_color="transparent", border_width=0)
        workflow.grid(row=1, column=0, padx=24, pady=(8, 6), sticky="ew")
        workflow.grid_columnconfigure(0, weight=1)
        stepper = ctk.CTkFrame(workflow, corner_radius=8, border_width=1)
        stepper.grid(row=0, column=0, sticky="ew")
        stepper.grid_columnconfigure(tuple(range(len(WORKFLOW_STEPS))), weight=1)
        for index, (step, _tab_name) in enumerate(WORKFLOW_STEPS):
            button = ctk.CTkButton(
                stepper,
                text=f"{index + 1}. {step}",
                command=lambda selected=step: self.select_workflow_step(selected),
                height=30,
                corner_radius=7,
                font=("Arial", 12, "bold"),
            )
            button.grid(
                row=0,
                column=index,
                padx=5,
                pady=5,
                sticky="ew",
            )
            self._workflow_buttons[step] = button
        self._checks_button = ctk.CTkButton(
            stepper,
            text="Checks",
            command=lambda: self.select_tab("Checks"),
            height=24,
            corner_radius=7,
            font=("Arial", 11, "bold"),
        )
        self._checks_button.grid(
            row=1,
            column=len(WORKFLOW_STEPS) - 1,
            padx=5,
            pady=(0, 5),
            sticky="ew",
        )

    def _update_workflow_stepper(self, active_step: str) -> None:
        """Style the global workflow stepper with one active step."""
        self._active_workflow_step = active_step
        checks_active = active_step == "Checks"
        for step, button in self._workflow_buttons.items():
            is_active = step == active_step
            button.configure(
                fg_color="#12313a" if is_active else "#071018",
                hover_color="#173e49" if is_active else "#102232",
                border_width=1,
                border_color="#2ac6d6" if is_active else "#315a6d",
                text_color="#e7faff" if is_active else "#c7d8e8",
            )
        if self._checks_button is not None:
            self._checks_button.configure(
                fg_color="#182735" if checks_active else "#071018",
                hover_color="#22394a" if checks_active else "#102232",
                border_width=1,
                border_color="#6f91aa" if checks_active else "#315a6d",
                text_color="#e7faff" if checks_active else "#c7d8e8",
            )

    def select_workflow_step(self, step: str) -> None:
        """Navigate to a workflow step."""
        tab_name = dict(WORKFLOW_STEPS)[step]
        self.select_tab(tab_name)
        self._update_workflow_stepper(step)
        if step == "Project Setup":
            self.project_tab.show_project_setup_section()
        elif step == "Training Data":
            self.project_tab.show_training_data_section()

    def _sync_workflow_from_selected_tab(self) -> None:
        """Keep the workflow stepper aligned with direct tab navigation."""
        selected = self.tabs.get()
        if selected == "Project":
            self._update_workflow_stepper(
                self.project_tab.current_project_workflow_step()
            )
        elif selected == "Train":
            self._update_workflow_stepper("Training")
        elif selected == "Test":
            self._update_workflow_stepper("Testing")
        elif selected == "Export":
            self._update_workflow_stepper("Export")
        elif selected == "Checks":
            self._update_workflow_stepper("Checks")

    def set_appearance_mode(self, mode: str) -> None:
        """Set the application appearance mode.

        Args:
            mode (str): Appearance mode name, expected to be ``Dark`` or
                ``Light``.
        """
        ctk.set_appearance_mode(mode)

    def select_tab(self, name: str) -> None:
        """Select one of the main application tabs."""
        self.tabs.set(name)
        self._sync_workflow_from_selected_tab()

    def set_project(self, project: WakeWordProject) -> None:
        """Set the current project.

        Args:
            project (WakeWordProject): Current project.
        """
        self.current_project = project
        self.project_label.configure(
            text=(
                f"Project: {project.model_name} | "
                f"Phrase: {project.wake_phrase} | "
                f"Workspace: {project.workspace_dir}"
            )
        )
        if hasattr(self, "test_tab"):
            self.test_tab.refresh_from_project()
        if hasattr(self, "export_tab"):
            self.export_tab.refresh_from_project()

    def clear_project(self) -> None:
        """Clear the current project selection."""
        self.current_project = None
        self.project_label.configure(text="No project loaded.")
        if hasattr(self, "test_tab"):
            self.test_tab.refresh_from_project()
        if hasattr(self, "export_tab"):
            self.export_tab.refresh_from_project()

    def get_project(self) -> WakeWordProject | None:
        """Return the current project.

        Returns:
            WakeWordProject | None: Current project, if any.
        """
        return self.current_project


def _resolve_theme_path(
    *,
    themes_dir: Path = THEMES_DIR,
    selection_path: Path = THEME_SELECTION_PATH,
    fallback_path: Path = DEFAULT_THEME_PATH,
) -> Path:
    """Resolve the CustomTkinter theme file to load.

    The loader prefers a theme name stored in ``set_theme.txt`` when it maps
    to an existing JSON file in the themes directory. If the selection file is
    missing, blank, or invalid, the bundled NightTrain theme is used.

    Args:
        themes_dir (Path): Directory containing theme JSON files.
        selection_path (Path): Path to the text file with the desired theme.
        fallback_path (Path): Theme file used when selection is unavailable.

    Returns:
        Path: Path to the theme JSON file that should be loaded.
    """
    return _resolve_theme_settings(
        themes_dir=themes_dir,
        selection_path=selection_path,
        fallback_path=fallback_path,
    ).theme_path


def _resolve_theme_settings(
    *,
    themes_dir: Path = THEMES_DIR,
    selection_path: Path = THEME_SELECTION_PATH,
    fallback_path: Path = DEFAULT_THEME_PATH,
    fallback_mode: str = DEFAULT_APPEARANCE_MODE,
) -> ThemeSettings:
    """Resolve the CustomTkinter theme file and appearance mode.

    ``set_theme.txt`` uses ``<theme-name>:<mode>``. The parser also accepts
    the previous one-part format and theme names with a ``.json`` suffix.

    Args:
        themes_dir (Path): Directory containing theme JSON files.
        selection_path (Path): Path to the theme and mode selection file.
        fallback_path (Path): Theme file used when selection is unavailable.
        fallback_mode (str): Appearance mode used when selection is invalid.

    Returns:
        ThemeSettings: Theme file and appearance mode to apply.
    """
    theme_path = fallback_path
    appearance_mode = fallback_mode
    if selection_path.exists():
        raw_value = selection_path.read_text(encoding="utf-8").strip()
        theme_name, separator, mode_name = raw_value.partition(":")
        theme_name = theme_name.strip()
        if theme_name.lower().endswith(".json"):
            theme_name = theme_name[:-5]
        if theme_name:
            candidate = themes_dir / f"{theme_name}.json"
            if candidate.exists():
                theme_path = candidate
        if separator:
            normalized_mode = mode_name.strip().lower()
            if normalized_mode == "dark":
                appearance_mode = "Dark"
            elif normalized_mode == "light":
                appearance_mode = "Light"
    return ThemeSettings(theme_path=theme_path, appearance_mode=appearance_mode)
