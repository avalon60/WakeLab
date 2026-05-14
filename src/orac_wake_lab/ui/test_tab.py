"""File-based model test tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Tests trained wake-word models against selected WAV files.

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog
import threading

import customtkinter as ctk

from orac_wake_lab.services import model_tester


class TestTab(ctk.CTkFrame):
    """Test trained ONNX models against WAV files."""

    def __init__(self, master: ctk.CTkBaseClass, app: object) -> None:
        """Create the Test tab."""
        super().__init__(master)
        self.app = app
        self.model_path_var = ctk.StringVar(value="")
        self.wav_path_var = ctk.StringVar(value="")
        self.threshold_var = ctk.StringVar(value="0.75")
        self.status_var = ctk.StringVar(value="Select a WAV file to test.")
        self.output_text = ""
        self._build()
        self.refresh_from_project()

    def _build(self) -> None:
        intro = (
            "Test a trained ONNX wake-word model against a WAV file. "
            "Live microphone testing and threshold tuning can be added next; "
            "this file test is the first diagnostic path."
        )
        ctk.CTkLabel(
            self,
            text=intro,
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", padx=12, pady=12)

        form = ctk.CTkFrame(self)
        form.pack(fill="x", padx=12, pady=(0, 12))
        self._add_path_row(
            form,
            "Model",
            self.model_path_var,
            self.browse_model,
            0,
        )
        self._add_path_row(form, "WAV", self.wav_path_var, self.browse_wav, 1)

        ctk.CTkLabel(form, text="Threshold").grid(
            row=2,
            column=0,
            sticky="w",
            padx=6,
            pady=6,
        )
        ctk.CTkEntry(form, textvariable=self.threshold_var, width=120).grid(
            row=2,
            column=1,
            sticky="w",
            padx=6,
            pady=6,
        )
        form.columnconfigure(1, weight=1)

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.pack(anchor="w", padx=12, pady=(0, 12))
        ctk.CTkButton(
            button_row,
            text="Refresh Model",
            command=self.refresh_from_project,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            button_row,
            text="Run WAV Test",
            command=self.run_wav_test,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            button_row,
            text="Copy To Clipboard",
            command=self.copy_output_to_clipboard,
        ).pack(side="left", padx=6)

        ctk.CTkLabel(self, textvariable=self.status_var).pack(
            anchor="w",
            padx=12,
        )
        self.output = ctk.CTkTextbox(self, wrap="word", height=260)
        self.output.pack(fill="both", expand=True, padx=12, pady=12)
        self.output.configure(state="disabled")

    def _add_path_row(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: ctk.StringVar,
        command: object,
        row: int,
    ) -> None:
        ctk.CTkLabel(parent, text=label).grid(
            row=row,
            column=0,
            sticky="w",
            padx=6,
            pady=6,
        )
        ctk.CTkEntry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=6,
            pady=6,
        )
        ctk.CTkButton(parent, text="Browse", command=command).grid(
            row=row,
            column=2,
            padx=6,
            pady=6,
        )

    def refresh_from_project(self) -> None:
        """Fill the model path from the current project."""
        project = self.app.get_project()
        if project is None:
            self.model_path_var.set("")
            return
        self.model_path_var.set(str(model_tester.default_model_path(project)))

    def browse_model(self) -> None:
        """Select an ONNX model file."""
        project = self.app.get_project()
        initial_dir = Path(self.model_path_var.get()).expanduser().parent
        if project is not None:
            initial_dir = project.openwakeword_output_dir
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
        if selected:
            self.model_path_var.set(selected)

    def browse_wav(self) -> None:
        """Select a WAV file to test."""
        project = self.app.get_project()
        wav_path_value = self.wav_path_var.get().strip()
        initial_dir = Path(wav_path_value).expanduser().parent
        if project is not None and (
            not wav_path_value or not initial_dir.exists()
        ):
            initial_dir = project.openwakeword_output_dir / project.model_name
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if selected:
            self.wav_path_var.set(selected)

    def run_wav_test(self) -> None:
        """Run the selected WAV through the selected model."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            self._set_output("Threshold must be a number between 0 and 1.\n")
            return
        model_path = Path(self.model_path_var.get()).expanduser()
        wav_path = Path(self.wav_path_var.get()).expanduser()
        self.status_var.set("Running WAV test...")

        def worker() -> None:
            try:
                result = model_tester.test_wav_file(
                    project,
                    model_path,
                    wav_path,
                    threshold,
                )
                message = result.render()
            except Exception as exc:
                message = f"WAV test failed: {exc}\n"
            self.after(0, self._set_output, message)

        threading.Thread(target=worker, daemon=True).start()

    def _set_output(self, text: str) -> None:
        self.output_text = text
        self.status_var.set("WAV test finished.")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def copy_output_to_clipboard(self) -> None:
        """Copy the Test tab output to the system clipboard."""
        self.clipboard_clear()
        self.clipboard_append(self.output_text)
        self.update_idletasks()
