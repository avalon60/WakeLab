"""openWakeWord training config generation for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Builds YAML configs for openWakeWord train.py.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.models.validation import ValidationResult


TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "n_samples": 1000,
        "n_samples_val": 1000,
        "tts_batch_size": 20,
        "augmentation_batch_size": 16,
        "augmentation_rounds": 1,
        "steps": 10000,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
        "layer_size": 32,
        "model_type": "dnn",
    },
    "balanced": {
        "n_samples": 10000,
        "n_samples_val": 2000,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "augmentation_rounds": 1,
        "steps": 50000,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
        "layer_size": 32,
        "model_type": "dnn",
    },
    "manual": {
        "n_samples": 1000,
        "n_samples_val": 1000,
        "tts_batch_size": 20,
        "augmentation_batch_size": 16,
        "augmentation_rounds": 1,
        "steps": 10000,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
        "layer_size": 32,
        "model_type": "dnn",
    },
}


def build_training_config(project: WakeWordProject) -> dict[str, Any]:
    """Build an openWakeWord ``train.py`` YAML config.

    The keys are based on ``examples/custom_model.yml`` and the lookups in
    ``openwakeword/train.py``.

    Args:
        project (WakeWordProject): Wake-word project settings.

    Returns:
        dict[str, Any]: YAML-serialisable config.
    """
    profile = TRAINING_PROFILES.get(
        project.profile,
        TRAINING_PROFILES["quick"],
    )
    feature_data_files = {
        name: str(path)
        for name, path in (project.negative_feature_data_files or {}).items()
    }
    batch_n_per_class = {
        "adversarial_negative": 50,
        "positive": 50,
    }
    for name in feature_data_files:
        batch_n_per_class[name] = 1024

    config: dict[str, Any] = {
        "model_name": project.model_name,
        "target_phrase": [project.wake_phrase],
        "custom_negative_phrases": project.custom_negative_phrases,
        "piper_sample_generator_path": str(
            project.piper_sample_generator_path
        ),
        "output_dir": str(project.openwakeword_output_dir),
        "rir_paths": [str(path) for path in project.rir_paths or []],
        "background_paths": [
            str(path) for path in project.background_paths or []
        ],
        "background_paths_duplication_rate": [
            1 for _ in project.background_paths or []
        ],
        "false_positive_validation_data_path": str(
            project.false_positive_validation_data_path
        ),
        "feature_data_files": feature_data_files,
        "batch_n_per_class": batch_n_per_class,
    }
    config.update(profile)
    return config


def validate_training_config_inputs(
    project: WakeWordProject,
) -> ValidationResult:
    """Validate training-specific config inputs.

    Args:
        project (WakeWordProject): Wake-word project settings.

    Returns:
        ValidationResult: Validation result for training inputs.
    """
    feature_files = project.negative_feature_data_files or {}
    if not feature_files:
        return ValidationResult(
            name="Negative feature data files",
            status="fail",
            message=(
                "At least one precomputed negative feature .npy file is "
                "required for openWakeWord training."
            ),
            blocks=["train"],
        )
    missing = [
        f"{name}: {path}"
        for name, path in feature_files.items()
        if not path.exists() or path.suffix != ".npy"
    ]
    if missing:
        return ValidationResult(
            name="Negative feature data files",
            status="fail",
            message="Missing or invalid .npy files: " + ", ".join(missing),
            blocks=["train"],
        )
    return ValidationResult(
        name="Negative feature data files",
        status="pass",
        message="Negative feature data files are configured.",
        blocks=[],
    )


def write_training_config(project: WakeWordProject) -> Path:
    """Write the generated training config to disk.

    Args:
        project (WakeWordProject): Wake-word project settings.

    Returns:
        Path: Written YAML path.
    """
    project.config_dir.mkdir(parents=True, exist_ok=True)
    config = build_training_config(project)
    project.training_config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    return project.training_config_path
