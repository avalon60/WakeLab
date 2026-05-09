"""Orac export service for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Copies trained wake models and creates Orac config snippets.

from __future__ import annotations

import shutil
from pathlib import Path

from orac_wake_lab.models.project import WakeWordProject


def build_smoke_test_command(project: WakeWordProject) -> str:
    """Build the Orac wake-word smoke-test command.

    Args:
        project (WakeWordProject): Project containing the Orac repo path.

    Returns:
        str: Shell command for display.
    """
    return (
        f"cd {project.orac_repo}\n"
        "PYTHONPATH=src poetry run python -m orac_voice.voice_loop_local "
        "--voice-session --activation-mode openwakeword"
    )


def find_generated_models(project: WakeWordProject) -> list[Path]:
    """Find generated ONNX/TFLite models and mirror them to project models.

    openWakeWord exports final models to ``output_dir`` using
    ``<model_name>.onnx`` and optionally ``<model_name>.tflite``.

    Args:
        project (WakeWordProject): Project to inspect.

    Returns:
        list[Path]: Mirrored model paths under the project ``models`` dir.
    """
    project.models_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        project.openwakeword_output_dir / f"{project.model_name}.onnx",
        project.openwakeword_output_dir / f"{project.model_name}.tflite",
    ]
    mirrored: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            target = project.models_dir / candidate.name
            if candidate.resolve() != target.resolve():
                shutil.copy2(candidate, target)
            mirrored.append(target)
    return mirrored


def build_orac_config_snippet(model_file: Path) -> str:
    """Build a candidate Orac voice config snippet.

    Args:
        model_file (Path): Exported model file path.

    Returns:
        str: INI snippet.
    """
    return (
        "[voice]\n"
        "activation_mode = openwakeword\n"
        "wake_engine = openwakeword\n"
        "openwakeword_model_paths = "
        f"${{ORAC_HOME}}/var/models/wake/{model_file.name}\n"
        "openwakeword_model_names =\n"
        "openwakeword_threshold = 0.75\n"
        "openwakeword_inference_framework = auto\n"
    )


def export_model_to_orac(
    project: WakeWordProject,
    model_file: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, str]:
    """Copy a trained model into Orac and write a candidate config.

    Args:
        project (WakeWordProject): Source project.
        model_file (Path): Model file to export.
        overwrite (bool): Whether to replace an existing target model.

    Returns:
        tuple[Path, Path, str]: Exported model path, candidate config path,
        and smoke-test command.

    Raises:
        FileNotFoundError: If the selected model does not exist.
        ValueError: If the selected model has an unsupported extension.
    """
    model_file = model_file.expanduser()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")
    if model_file.suffix not in {".onnx", ".tflite"}:
        raise ValueError("Only .onnx and .tflite models can be exported.")

    target_dir = project.orac_repo / "var" / "models" / "wake"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / model_file.name
    if target_file.exists() and not overwrite:
        raise FileExistsError(
            f"Target model already exists: {target_file}"
        )
    shutil.copy2(model_file, target_file)

    snippet = build_orac_config_snippet(target_file)
    project.config_dir.mkdir(parents=True, exist_ok=True)
    project.orac_candidate_config_path.write_text(snippet, encoding="utf-8")
    return (
        target_file,
        project.orac_candidate_config_path,
        build_smoke_test_command(project),
    )
