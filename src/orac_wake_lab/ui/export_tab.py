"""Model export tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Exports generated wake-word models to selected target directories.

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox
from tkinter import filedialog

import customtkinter as ctk

from orac_wake_lab.services.orac_export import export_model_to_directory
from orac_wake_lab.services.orac_export import find_generated_models


class ExportTab(ctk.CTkFrame):
    """Export generated models to a selected runtime target."""

    def __init__(self, master: ctk.CTkBaseClass, app: object) -> None:
        """Create the export tab."""
        super().__init__(master)
        self.app = app
        self.model_path_var = ctk.StringVar(value="")
        self.target_dir_var = ctk.StringVar(value="")
        self.status_var = ctk.StringVar(value="No model exported.")
        self.embed_onnx_external_data_var = ctk.BooleanVar(value=False)
        self.overwrite_var = ctk.BooleanVar(value=False)
        self._build()
        self.model_path_var.trace_add("write", self._sync_export_options)
        self._sync_export_options()

    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            self,
            text="Refresh Generated Models",
            command=self.refresh_models,
        ).grid(row=0, column=0, padx=12, pady=12, sticky="w")
        ctk.CTkLabel(self, text="Model").grid(
            row=1,
            column=0,
            padx=12,
            pady=(6, 0),
            sticky="w",
        )
        ctk.CTkEntry(self, textvariable=self.model_path_var).grid(
            row=2,
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
        ).grid(row=2, column=2, padx=12, pady=6)
        ctk.CTkLabel(self, text="Target directory").grid(
            row=3,
            column=0,
            padx=12,
            pady=(6, 0),
            sticky="w",
        )
        ctk.CTkEntry(self, textvariable=self.target_dir_var).grid(
            row=4,
            column=0,
            columnspan=2,
            padx=12,
            pady=6,
            sticky="ew",
        )
        ctk.CTkButton(
            self,
            text="Browse",
            command=self.browse_target_dir,
        ).grid(row=4, column=2, padx=12, pady=6)
        ctk.CTkButton(
            self,
            text="Export Model",
            command=self.export_model,
        ).grid(row=5, column=0, padx=12, pady=12, sticky="w")
        self.output = ctk.CTkTextbox(self, wrap="word", height=260)
        self.output.grid(
            row=6,
            column=0,
            columnspan=3,
            padx=12,
            pady=12,
            sticky="nsew",
        )
        self.grid_rowconfigure(6, weight=1)
        self.output.insert(
            "end",
            "Select a generated .onnx or .tflite model and export it to any "
            "runtime directory. ONNX external-data sidecars are copied when "
            "present, and ONNX files can optionally be embedded into a "
            "single file.",
        )
        self.output.configure(state="disabled")
        self.embed_onnx_checkbox = ctk.CTkCheckBox(
            self,
            text="Embed ONNX external data into a single file",
            variable=self.embed_onnx_external_data_var,
        )
        self.embed_onnx_checkbox.grid(
            row=7,
            column=0,
            columnspan=3,
            padx=12,
            pady=6,
            sticky="w",
        )
        ctk.CTkCheckBox(
            self,
            text="Allow overwrite of existing exported model",
            variable=self.overwrite_var,
        ).grid(row=8, column=0, columnspan=3, padx=12, pady=6, sticky="w")
        ctk.CTkLabel(self, textvariable=self.status_var).grid(
            row=9,
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
            self.status_var.set("Create or open a project first.")
            return
        self._set_default_target_dir(project)
        models = find_generated_models(project)
        if not models:
            self.model_path_var.set("")
            self._sync_export_options()
            self.status_var.set(
                "No generated .onnx or .tflite models found for this project."
            )
            return
        self.model_path_var.set(str(models[-1]))
        self._sync_export_options()
        self.status_var.set(f"Found {len(models)} model file(s).")

    def refresh_from_project(self) -> None:
        """Refresh export defaults from the current project."""
        project = self.app.get_project()
        if project is None:
            self.model_path_var.set("")
            self.target_dir_var.set("")
            self.status_var.set("No model exported.")
            return
        self._set_default_target_dir(project)
        models = find_generated_models(project)
        if models:
            self.model_path_var.set(str(models[-1]))
            self._sync_export_options()
            self.status_var.set(f"Found {len(models)} model file(s).")
        else:
            self.model_path_var.set("")
            self._sync_export_options()
            self.status_var.set(
                "No generated .onnx or .tflite models found for this project."
            )

    def browse_model(self) -> None:
        """Select a model file manually."""
        project = self.app.get_project()
        model_path_value = self.model_path_var.get().strip()
        initial_dir = Path(model_path_value).expanduser().parent
        if project is not None and (
            not model_path_value or not initial_dir.exists()
        ):
            initial_dir = (
                project.models_dir
                if project.models_dir.exists()
                else project.openwakeword_output_dir
            )
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            filetypes=[
                ("openWakeWord models", "*.onnx *.tflite"),
                ("All files", "*.*"),
            ]
        )
        if selected:
            self.model_path_var.set(selected)
            self._sync_export_options()

    def browse_target_dir(self) -> None:
        """Select an export target directory manually."""
        project = self.app.get_project()
        target_dir_value = self.target_dir_var.get().strip()
        initial_dir = Path(target_dir_value).expanduser()
        if not initial_dir.exists() and project is not None:
            initial_dir = project.export_dir
        selected = filedialog.askdirectory(initialdir=str(initial_dir))
        if selected:
            self.target_dir_var.set(selected)

    def export_model(self) -> None:
        """Export the selected model to the selected target directory."""
        model_path = Path(self.model_path_var.get().strip()).expanduser()
        target_dir = Path(self.target_dir_var.get().strip()).expanduser()
        if not self.model_path_var.get().strip():
            self.status_var.set("Select a model to export.")
            return
        if not self.target_dir_var.get().strip():
            self.status_var.set("Select a target directory.")
            return
        target_path = target_dir / model_path.name
        if target_path.exists() and self.overwrite_var.get():
            confirmed = messagebox.askyesno(
                "Confirm overwrite",
                f"Replace existing exported model?\n\n{target_path}",
            )
            if not confirmed:
                self.status_var.set("Export cancelled.")
                return
        try:
            target = export_model_to_directory(
                model_path,
                target_dir,
                overwrite=self.overwrite_var.get(),
                embed_onnx_external_data=(
                    self.embed_onnx_external_data_var.get()
                    and model_path.suffix.lower() == ".onnx"
                ),
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            return
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        output = f"Exported model:\n{target}\n"
        sidecar = target.with_name(f"{target.name}.data")
        if (
            target.suffix == ".onnx"
            and not self.embed_onnx_external_data_var.get()
            and sidecar.exists()
        ):
            output += f"\nExported ONNX sidecar:\n{sidecar}\n"
        elif target.suffix == ".onnx" and self.embed_onnx_external_data_var.get():
            output += "\nEmbedded ONNX external data into the exported model.\n"
        self.output.insert("end", output)
        self.output.configure(state="disabled")
        self.status_var.set(f"Exported model: {target}")
        self._sync_export_options()

    def _set_default_target_dir(self, project: object) -> None:
        """Set the default target directory when none is selected."""
        if self.target_dir_var.get().strip():
            return
        self.target_dir_var.set(str(project.export_dir))

    def _sync_export_options(self, *_args: object) -> None:
        """Enable export options that apply to the current model selection."""
        model_path = Path(self.model_path_var.get().strip()).expanduser()
        is_onnx = model_path.suffix.lower() == ".onnx"
        self.embed_onnx_checkbox.configure(
            state="normal" if is_onnx else "disabled"
        )
        if not is_onnx:
            self.embed_onnx_external_data_var.set(False)
