"""Dependency checks tab for WakeLab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Displays dependency and path check results.

from __future__ import annotations

import threading

import customtkinter as ctk

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import ValidationResult
from orac_wake_lab.services.dependency_checks import (
    run_dependency_checks,
)


class ChecksTab(ctk.CTkFrame):
    """Dependency checks tab."""

    def __init__(self, master: ctk.CTkBaseClass, app: object) -> None:
        """Create the checks tab."""
        super().__init__(master)
        self.app = app
        self.output = ctk.CTkTextbox(self, wrap="word")
        self.output.pack(fill="both", expand=True, padx=12, pady=12)
        button_row = ctk.CTkFrame(self, fg_color="transparent", border_width=0)
        button_row.pack(anchor="w", padx=12, pady=(0, 12))
        self.back_button = ctk.CTkButton(
            button_row,
            text="Back To Project",
            command=self.back_to_project,
        )
        self.back_button.pack(side="left", padx=(0, 8))
        self.run_button = ctk.CTkButton(
            button_row,
            text="Run Checks",
            command=self.run_checks,
        )
        self.run_button.pack(side="left", padx=(0, 8))
        self.copy_button = ctk.CTkButton(
            button_row,
            text="Copy To Clipboard",
            command=self.copy_output_to_clipboard,
        )
        self.copy_button.pack(side="left")

    def back_to_project(self) -> None:
        """Return to the Project Setup workflow step."""
        if hasattr(self.app, "select_workflow_step"):
            self.app.select_workflow_step("Project Setup")
            return
        if hasattr(self.app, "select_tab"):
            self.app.select_tab("Project")

    def run_checks(self) -> None:
        """Run dependency checks for the current project."""
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        project = self.app.get_project()
        if project is None:
            self.output.insert("end", "Create or open a project first.\n")
            self.output.configure(state="disabled")
            return
        self.output.insert("end", "Running checks...\n")
        self.output.configure(state="disabled")
        self.run_button.configure(state="disabled")
        threading.Thread(
            target=self._run_checks_worker,
            args=(project,),
            daemon=True,
        ).start()

    def _run_checks_worker(self, project: WakeWordProject) -> None:
        """Run checks in a worker thread."""
        results = run_dependency_checks(project)
        self.after(0, self._show_results, results)

    def _show_results(self, results: list[ValidationResult]) -> None:
        """Render check results on the GUI thread."""
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        for result in results:
            blocks = ", ".join(result.blocks) if result.blocks else "-"
            self.output.insert(
                "end",
                f"[{result.status.upper()}] {result.name}\n"
                f"  {result.message}\n"
                f"  Blocks: {blocks}\n\n",
            )
        self.output.configure(state="disabled")
        self.run_button.configure(state="normal")

    def copy_output_to_clipboard(self) -> None:
        """Copy the current checks output to the system clipboard."""
        text = self.output.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()
