"""Orac export service for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Copies trained wake models and creates Orac config snippets.

from __future__ import annotations

import importlib
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
            _copy_onnx_sidecar(candidate, target, overwrite=True)
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


def export_model_to_directory(
    model_file: Path,
    target_dir: Path,
    *,
    overwrite: bool = False,
    embed_onnx_external_data: bool = False,
) -> Path:
    """Copy a trained model into a selected target directory.

    Args:
        model_file (Path): Model file to export.
        target_dir (Path): Directory that should receive the model.
        overwrite (bool): Whether to replace an existing target model.
        embed_onnx_external_data (bool): Whether ONNX external data should
          be embedded into the ONNX file instead of copied beside it.

    Returns:
        Path: Exported model path.

    Raises:
        FileNotFoundError: If the selected model does not exist.
        ValueError: If the selected model has an unsupported extension.
        FileExistsError: If the target exists and overwrite is false.
    """
    model_file = model_file.expanduser()
    target_dir = target_dir.expanduser()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")
    if model_file.suffix not in {".onnx", ".tflite"}:
        raise ValueError("Only .onnx and .tflite models can be exported.")

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / model_file.name
    if target_file.exists() and not overwrite:
        raise FileExistsError(
            f"Target model already exists: {target_file}"
        )

    if model_file.suffix == ".onnx" and embed_onnx_external_data:
        target_sidecar = _onnx_sidecar_path(target_file)
        if target_sidecar.exists() and not overwrite:
            raise FileExistsError(
                f"Target ONNX external data file already exists: {target_sidecar}"
            )
        if overwrite and target_sidecar.exists():
            target_sidecar.unlink()
        embed_onnx_external_data_file(model_file, target_file)
        return target_file

    sidecar = _onnx_sidecar_path(model_file)
    target_sidecar = _onnx_sidecar_path(target_file)
    if sidecar is not None and not sidecar.exists():
        raise FileNotFoundError(
            f"ONNX external data file does not exist: {sidecar}"
        )
    if (
        sidecar is not None
        and target_sidecar.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Target ONNX external data file already exists: {target_sidecar}"
        )

    shutil.copy2(model_file, target_file)
    _copy_onnx_sidecar(model_file, target_file, overwrite=overwrite)
    return target_file


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
    target_dir = project.orac_repo / "var" / "models" / "wake"
    target_file = export_model_to_directory(
        model_file,
        target_dir,
        overwrite=overwrite,
    )
    snippet = build_orac_config_snippet(target_file)
    project.config_dir.mkdir(parents=True, exist_ok=True)
    project.orac_candidate_config_path.write_text(snippet, encoding="utf-8")
    return (
        target_file,
        project.orac_candidate_config_path,
        build_smoke_test_command(project),
    )


def _onnx_sidecar_path(model_file: Path) -> Path | None:
    """Return the external-data sidecar path for an ONNX model."""
    if model_file.suffix != ".onnx":
        return None
    return model_file.with_name(f"{model_file.name}.data")


def _copy_onnx_sidecar(
    source_model: Path,
    target_model: Path,
    *,
    overwrite: bool,
) -> None:
    """Copy an ONNX external-data sidecar when one exists."""
    source_sidecar = _onnx_sidecar_path(source_model)
    target_sidecar = _onnx_sidecar_path(target_model)
    if source_sidecar is None or target_sidecar is None:
        return
    if not source_sidecar.exists():
        return
    if target_sidecar.exists() and not overwrite:
        raise FileExistsError(
            f"Target ONNX external data file already exists: {target_sidecar}"
        )
    shutil.copy2(source_sidecar, target_sidecar)


def embed_onnx_external_data_file(
    source_onnx: Path,
    target_onnx: Path,
) -> None:
    """Embed ONNX external data into a single self-contained model file.

    Args:
        source_onnx (Path): Source ONNX model that may reference external data.
        target_onnx (Path): Target ONNX path to create.

    Raises:
        FileNotFoundError: If the source ONNX file does not exist.
        RuntimeError: If the `onnx` package is unavailable.
    """
    if not source_onnx.exists():
        raise FileNotFoundError(f"Source ONNX file not found: {source_onnx}")

    onnx = _import_onnx_module()
    target_onnx.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source_onnx), load_external_data=True)
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, str(target_onnx))
    onnx.load(str(target_onnx), load_external_data=False)
    onnx.checker.check_model(str(target_onnx))


def _import_onnx_module() -> object:
    """Import the ONNX module on demand."""
    try:
        return importlib.import_module("onnx")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Embedding ONNX external data requires the 'onnx' package."
        ) from exc
