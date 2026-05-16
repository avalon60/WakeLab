"""File-based model test tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-14
# Description: Tests trained wake-word models against selected WAV files.

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from tkinter import filedialog
import threading

import customtkinter as ctk

from orac_wake_lab.services import model_tester
from orac_wake_lab.services import near_misses


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
        eval_row = ctk.CTkFrame(self, fg_color="transparent")
        eval_row.pack(anchor="w", padx=12, pady=(0, 12))
        eval_buttons = [
            ("Score Synthetic Positives", self.score_synthetic_positives),
            ("Score Real Positives", self.score_real_positives),
            ("Score Similar Phrases", self.score_near_misses),
            ("Score Negative Clips", self.score_negative_clips),
        ]
        for text, command in eval_buttons:
            ctk.CTkButton(
                eval_row,
                text=text,
                command=command,
            ).pack(side="left", padx=(0, 6))

        near_miss_help = (
            "Near-miss clips are similar phrases that should not activate the "
            "wake word, such as 'Hey Oracle', 'Hey Oreck', 'Hey Eric', "
            "'Oracle', or 'Orac'."
        )
        ctk.CTkLabel(
            self,
            text=near_miss_help,
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        near_miss_row = ctk.CTkFrame(self, fg_color="transparent")
        near_miss_row.pack(anchor="w", padx=12, pady=(0, 12))
        ctk.CTkButton(
            near_miss_row,
            text="Open Near-Misses Folder",
            command=self.open_near_misses_folder,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            near_miss_row,
            text="Import Near-Miss WAVs",
            command=self.import_near_miss_wavs,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            near_miss_row,
            text="Generate Synthetic Near-Miss Clips",
            command=self.generate_synthetic_near_misses,
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
        """Fill model and WAV defaults from the current project."""
        project = self.app.get_project()
        if project is None:
            self.model_path_var.set("")
            self.wav_path_var.set("")
            return
        project.near_miss_clips_dir.mkdir(parents=True, exist_ok=True)
        self.model_path_var.set(str(model_tester.default_model_path(project)))
        wav_path_value = self.wav_path_var.get().strip()
        if TestTab._should_reset_wav_path(project, wav_path_value):
            self.wav_path_var.set(TestTab._default_wav_path(project))

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
        if project is not None and TestTab._should_reset_wav_path(
            project,
            wav_path_value,
        ):
            initial_dir = TestTab._default_wav_browse_dir(project)
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

    @staticmethod
    def _should_reset_wav_path(project: object, wav_path_value: str) -> bool:
        """Return whether the current WAV path is stale for this project."""
        if not wav_path_value:
            return True

        wav_path = Path(wav_path_value).expanduser()
        if not wav_path.parent.exists():
            return True

        try:
            if wav_path.resolve().is_relative_to(project.workspace_dir.resolve()):
                return False
        except OSError:
            return True

        return TestTab._is_other_project_path(project, wav_path)

    @staticmethod
    def _is_other_project_path(project: object, path: Path) -> bool:
        """Return whether a path belongs to a different WakeLab project."""
        projects_root = project.workspace_dir.parent.resolve()
        try:
            resolved_path = path.resolve()
        except OSError:
            return False

        for ancestor in (resolved_path, *resolved_path.parents):
            if ancestor.parent != projects_root:
                continue
            if ancestor == project.workspace_dir.resolve():
                return False
            return (ancestor / "project.json").exists()
        return False

    @staticmethod
    def _default_wav_browse_dir(project: object) -> Path:
        """Return the best project-local WAV directory for manual tests."""
        synthetic_positive_dir = (
            project.openwakeword_output_dir
            / project.model_name
            / "positive_test"
        )
        if synthetic_positive_dir.exists():
            return synthetic_positive_dir
        return project.real_positive_clips_dir

    @staticmethod
    def _default_wav_path(project: object) -> str:
        """Return the preferred project-local WAV test file, if available."""
        for directory in (
            project.openwakeword_output_dir / project.model_name / "positive_test",
            project.real_positive_clips_dir,
            project.openwakeword_output_dir / project.model_name / "negative_test",
        ):
            if not directory.exists():
                continue
            wav_files = sorted(
                path for path in directory.glob("*.wav") if path.is_file()
            )
            if wav_files:
                return str(wav_files[0])
        return ""

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

    def score_synthetic_positives(self) -> None:
        """Score generated positive test clips."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        self._score_directory(
            "Synthetic positive test clips",
            project.openwakeword_output_dir / project.model_name / "positive_test",
        )

    def score_real_positives(self) -> None:
        """Score project-local real positive clips."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        self._score_directory("Real positive clips", project.real_positive_clips_dir)

    def score_near_misses(self) -> None:
        """Score similar phrases that should not wake Orac."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        self._score_directory(
            "Similar phrases that should not wake Orac",
            project.near_miss_clips_dir,
            empty_guidance=(
                "No near-miss WAV files were found. Add or generate recordings "
                "of similar phrases that should not activate the wake word, "
                "such as 'Hey Oracle', 'Hey Oreck', or 'Oracle'.\n"
            ),
        )

    def score_negative_clips(self) -> None:
        """Score generated negative test clips."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        self._score_directory(
            "Negative test clips",
            project.openwakeword_output_dir / project.model_name / "negative_test",
        )

    def open_near_misses_folder(self) -> None:
        """Open the project near-misses folder in the file manager."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        project.near_miss_clips_dir.mkdir(parents=True, exist_ok=True)
        try:
            _open_folder(project.near_miss_clips_dir)
        except Exception as exc:
            self._set_output(f"Unable to open near-misses folder: {exc}\n")

    def import_near_miss_wavs(self) -> None:
        """Import WAV files into the project near-misses folder."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        project.near_miss_clips_dir.mkdir(parents=True, exist_ok=True)
        selected = filedialog.askopenfilenames(
            initialdir=str(project.near_miss_clips_dir),
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            imported = near_misses.import_near_miss_wavs(
                project,
                [Path(item) for item in selected],
            )
        except Exception as exc:
            self._set_output(f"Near-miss import failed: {exc}\n")
            return
        self.status_var.set(
            f"Imported {len(imported)} near-miss WAV(s)."
        )
        self._set_output(
            "Imported near-miss WAV files:\n"
            + "\n".join(str(path) for path in imported)
            + "\n"
        )

    def generate_synthetic_near_misses(self) -> None:
        """Generate synthetic near-miss clips from the negative phrases."""
        project = self.app.get_project()
        if project is None:
            self._set_output("Create or open a project first.\n")
            return
        self.status_var.set("Generating synthetic near-miss clips...")

        def worker() -> None:
            try:
                generated = near_misses.generate_synthetic_near_miss_wavs(project)
                if generated:
                    message = (
                        "Generated synthetic near-miss clips:\n"
                        + "\n".join(str(path) for path in generated)
                        + "\n"
                    )
                else:
                    message = (
                        "No negative phrases were configured, so no synthetic "
                        "near-miss clips were generated.\n"
                    )
            except Exception as exc:
                message = f"Near-miss generation failed: {exc}\n"
            self.after(0, self._set_output, message)

        threading.Thread(target=worker, daemon=True).start()

    def _score_directory(
        self,
        label: str,
        directory: Path,
        *,
        empty_guidance: str | None = None,
    ) -> None:
        """Score all WAV files in a directory on a background thread."""
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
        wav_paths = sorted(path for path in directory.glob("*.wav") if path.is_file())
        if not wav_paths:
            if empty_guidance is not None:
                self._set_output(empty_guidance)
            else:
                self._set_output(f"No WAV files found in: {directory}\n")
            return
        self.status_var.set(f"Scoring {label}...")

        def worker() -> None:
            try:
                results = model_tester.test_wav_directory(
                    project,
                    model_path,
                    directory,
                    threshold,
                )
                message = model_tester.render_directory_results(
                    label,
                    directory,
                    results,
                )
            except Exception as exc:
                message = f"{label} scoring failed: {exc}\n"
            self.after(0, self._set_output, message)

        threading.Thread(target=worker, daemon=True).start()


def _open_folder(folder: Path) -> None:
    """Open a folder in the platform's file manager."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if platform.system() == "Windows":
        os.startfile(folder)  # type: ignore[attr-defined]
        return
    if platform.system() == "Darwin":
        subprocess.Popen(["open", str(folder)])
        return
    subprocess.Popen(["xdg-open", str(folder)])
