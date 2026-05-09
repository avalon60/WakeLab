"""Main CustomTkinter window for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Composes the Phase 1 Orac Wake Lab desktop UI.

from __future__ import annotations

import customtkinter as ctk

from orac_wake_lab import APP_NAME
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.ui.checks_tab import ChecksTab
from orac_wake_lab.ui.export_tab import ExportTab
from orac_wake_lab.ui.project_tab import ProjectTab
from orac_wake_lab.ui.test_tab import TestTab
from orac_wake_lab.ui.training_tab import TrainingTab


class OracWakeLabApp(ctk.CTk):
    """Main Orac Wake Lab application window."""

    def __init__(self) -> None:
        """Create the main application window."""
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1100x760")
        self.current_project: WakeWordProject | None = None
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.project_label = ctk.CTkLabel(
            self,
            text="No project loaded.",
            anchor="w",
        )
        self.project_label.grid(
            row=0,
            column=0,
            padx=12,
            pady=8,
            sticky="ew",
        )
        tabs = ctk.CTkTabview(self)
        tabs.grid(row=1, column=0, padx=12, pady=12, sticky="nsew")
        for name in ["Project", "Checks", "Train", "Test", "Export"]:
            tabs.add(name)
        ProjectTab(tabs.tab("Project"), self).pack(fill="both", expand=True)
        ChecksTab(tabs.tab("Checks"), self).pack(fill="both", expand=True)
        TrainingTab(tabs.tab("Train"), self).pack(fill="both", expand=True)
        TestTab(tabs.tab("Test"), self).pack(fill="both", expand=True)
        ExportTab(tabs.tab("Export"), self).pack(fill="both", expand=True)

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

    def get_project(self) -> WakeWordProject | None:
        """Return the current project.

        Returns:
            WakeWordProject | None: Current project, if any.
        """
        return self.current_project
