"""Orac export tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Exports generated wake-word models into Orac safely.

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox
from tkinter import filedialog

import customtkinter as ctk

from orac_wake_lab.services.orac_export import export_model_to_orac
from orac_wake_lab.services.orac_export import find_generated_models


class ExportTab(ctk.CTkFrame):
    """Export generated models to Orac."""

    def __init__(self, master: ctk.CTkBaseClass, app: object) -> None:
        """Create the export tab."""
        super().__init__(master)
        self.app = app
        self.model_path_var = ctk.StringVar(value="")
        self.status_var = ctk.StringVar(value="No model exported.")
        self.overwrite_var = ctk.BooleanVar(value=False)
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            self,
            text="Refresh Generated Models",
            command=self.refresh_models,
        ).grid(row=0, column=0, padx=12, pady=12, sticky="w")
        ctk.CTkEntry(self, textvariable=self.model_path_var).grid(
            row=1,
            column=0,
            columnspan=2,
            padx=12,
            pady=6,
            sticky="ew",
        )
        ctk.CTkButton(
            self,
            text="Browse",
            command=self.browse_model,
        ).grid(row=1, column=2, padx=12, pady=6)
        ctk.CTkButton(
            self,
            text="Export To Orac",
            command=self.export_model,
        ).grid(row=2, column=0, padx=12, pady=12, sticky="w")
        self.output = ctk.CTkTextbox(self, wrap="word", height=260)
        self.output.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=12,
            pady=12,
            sticky="nsew",
        )
        self.grid_rowconfigure(3, weight=1)
        self.output.insert("end", "Create a project to view smoke test.")
        self.output.configure(state="disabled")
        ctk.CTkCheckBox(
            self,
            text="Allow overwrite of existing Orac model",
            variable=self.overwrite_var,
        ).grid(row=4, column=0, columnspan=3, padx=12, pady=6, sticky="w")
        ctk.CTkLabel(self, textvariable=self.status_var).grid(
            row=5,
            column=0,
            columnspan=3,
            padx=12,
            pady=8,
            sticky="w",
        )

    def refresh_models(self) -> None:
        """Refresh generated model detection."""
        project = self.app.get_project()
        if project is None:
            self.status_var.set("Create a project first.")
            return
        models = find_generated_models(project)
        if not models:
            self.status_var.set("No .onnx or .tflite models found.")
            return
        self.model_path_var.set(str(models[-1]))
        self.status_var.set(f"Found {len(models)} model file(s).")

    def browse_model(self) -> None:
        """Select a model file manually."""
        selected = filedialog.askopenfilename(
            filetypes=[
                ("openWakeWord models", "*.onnx *.tflite"),
                ("All files", "*.*"),
            ]
        )
        if selected:
            self.model_path_var.set(selected)

    def export_model(self) -> None:
        """Export the selected model to Orac."""
        project = self.app.get_project()
        if project is None:
            self.status_var.set("Create a project first.")
            return
        model_path = Path(self.model_path_var.get())
        target_path = (
            project.orac_repo / "var" / "models" / "wake" / model_path.name
        )
        if target_path.exists() and self.overwrite_var.get():
            confirmed = messagebox.askyesno(
                "Confirm overwrite",
                f"Replace existing Orac wake model?\n\n{target_path}",
            )
            if not confirmed:
                self.status_var.set("Export cancelled.")
                return
        try:
            target, config_path, smoke = export_model_to_orac(
                project,
                model_path,
                overwrite=self.overwrite_var.get(),
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            return
        snippet = config_path.read_text(encoding="utf-8")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert(
            "end",
            f"Exported model:\n{target}\n\n"
            f"Candidate config:\n{snippet}\n"
            f"Smoke test:\n{smoke}\n",
        )
        self.output.configure(state="disabled")
        self.status_var.set(f"Candidate config written: {config_path}")
