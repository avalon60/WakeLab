"""Main CustomTkinter window for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Composes the Phase 1 Orac Wake Lab desktop UI.

from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from orac_wake_lab import APP_NAME
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.ui.checks_tab import ChecksTab
from orac_wake_lab.ui.export_tab import ExportTab
from orac_wake_lab.ui.project_tab import ProjectTab
from orac_wake_lab.ui.test_tab import TestTab
from orac_wake_lab.ui.training_tab import TrainingTab


THEME_PATH = (
    Path(__file__).resolve().parents[1] / "themes" / "NightTrain.json"
)
DEFAULT_APPEARANCE_MODE = "Dark"
APPEARANCE_MODES = ("Dark", "Light")


class OracWakeLabApp(ctk.CTk):
    """Main Orac Wake Lab application window."""

    def __init__(self) -> None:
        """Create the main application window."""
        ctk.set_default_color_theme(str(THEME_PATH))
        ctk.set_appearance_mode(DEFAULT_APPEARANCE_MODE)
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1100x760")
        self.current_project: WakeWordProject | None = None
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

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

        tabs = ctk.CTkTabview(self)
        tabs.grid(row=1, column=0, padx=12, pady=12, sticky="nsew")
        for name in ["Project", "Checks", "Train", "Test", "Export"]:
            tabs.add(name)
        self.project_tab = ProjectTab(tabs.tab("Project"), self)
        self.checks_tab = ChecksTab(tabs.tab("Checks"), self)
        self.training_tab = TrainingTab(tabs.tab("Train"), self)
        self.test_tab = TestTab(tabs.tab("Test"), self)
        self.export_tab = ExportTab(tabs.tab("Export"), self)
        self.project_tab.pack(fill="both", expand=True)
        self.checks_tab.pack(fill="both", expand=True)
        self.training_tab.pack(fill="both", expand=True)
        self.test_tab.pack(fill="both", expand=True)
        self.export_tab.pack(fill="both", expand=True)

    def set_appearance_mode(self, mode: str) -> None:
        """Set the application appearance mode.

        Args:
            mode (str): Appearance mode name, expected to be ``Dark`` or
                ``Light``.
        """
        ctk.set_appearance_mode(mode)

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
