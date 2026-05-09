"""Project data model for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Defines persisted wake-word project settings.

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from orac_wake_lab.services import wake_lab_home


DEFAULT_OPENWAKEWORD_REPO = wake_lab_home.detect_openwakeword_repo()
DEFAULT_ORAC_REPO = wake_lab_home.detect_orac_repo()
DEFAULT_WORKSPACE_ROOT = wake_lab_home.get_projects_root()


@dataclass
class WakeWordProject:
    """Represent an Orac Wake Lab wake-word project.

    Args:
        wake_phrase (str): Wake phrase to train.
        model_name (str): Filesystem/model-safe model name.
        workspace_dir (Path): Project workspace directory.
        openwakeword_repo (Path): Local openWakeWord repository path.
        orac_repo (Path): Local Orac repository path.
        piper_sample_generator_path (Path): Piper sample generator path.
        background_paths (list[Path]): Background audio directories.
        rir_paths (list[Path]): Room impulse response directories.
        negative_feature_data_files (dict[str, Path]): Precomputed negative
          feature files used by openWakeWord training.
        false_positive_validation_data_path (Path): Validation feature path.
        custom_negative_phrases (list[str]): Extra negative phrases.
        profile (str): Training profile name.
    """

    wake_phrase: str
    model_name: str
    workspace_dir: Path
    openwakeword_repo: Path = DEFAULT_OPENWAKEWORD_REPO
    orac_repo: Path = DEFAULT_ORAC_REPO
    piper_sample_generator_path: Path = Path("")
    background_paths: list[Path] | None = None
    rir_paths: list[Path] | None = None
    negative_feature_data_files: dict[str, Path] | None = None
    false_positive_validation_data_path: Path = Path("")
    custom_negative_phrases: list[str] | None = None
    profile: str = "quick"

    def __post_init__(self) -> None:
        """Normalise mutable defaults and paths."""
        self.workspace_dir = self.workspace_dir.expanduser()
        self.openwakeword_repo = self.openwakeword_repo.expanduser()
        self.orac_repo = self.orac_repo.expanduser()
        self.piper_sample_generator_path = (
            self.piper_sample_generator_path.expanduser()
        )
        self.false_positive_validation_data_path = (
            self.false_positive_validation_data_path.expanduser()
        )
        self.background_paths = [
            path.expanduser() for path in (self.background_paths or [])
        ]
        self.rir_paths = [path.expanduser() for path in (self.rir_paths or [])]
        self.negative_feature_data_files = {
            name: path.expanduser()
            for name, path in (self.negative_feature_data_files or {}).items()
        }
        self.custom_negative_phrases = self.custom_negative_phrases or []

    @property
    def config_dir(self) -> Path:
        """Return the project config directory."""
        return self.workspace_dir / "config"

    @property
    def openwakeword_output_dir(self) -> Path:
        """Return the openWakeWord output directory."""
        return self.workspace_dir / "openwakeword_output"

    @property
    def models_dir(self) -> Path:
        """Return the project model mirror directory."""
        return self.workspace_dir / "models"

    @property
    def logs_dir(self) -> Path:
        """Return the project logs directory."""
        return self.workspace_dir / "logs"

    @property
    def export_dir(self) -> Path:
        """Return the project export directory."""
        return self.workspace_dir / "export"

    @property
    def training_config_path(self) -> Path:
        """Return the generated training config path."""
        return self.config_dir / "training.yaml"

    @property
    def orac_candidate_config_path(self) -> Path:
        """Return the candidate Orac config snippet path."""
        return self.config_dir / "orac.ini.candidate"

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        Returns:
            dict[str, Any]: Project data with paths converted to strings.
        """
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, list):
                data[key] = [
                    str(item) if isinstance(item, Path) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                data[key] = {
                    item_key: str(item_value)
                    if isinstance(item_value, Path)
                    else item_value
                    for item_key, item_value in value.items()
                }
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "WakeWordProject":
        """Build a project from persisted JSON data.

        Args:
            data (dict[str, Any]): Persisted project data.

        Returns:
            WakeWordProject: Rehydrated project.
        """
        path_fields = {
            "workspace_dir",
            "openwakeword_repo",
            "orac_repo",
            "piper_sample_generator_path",
            "false_positive_validation_data_path",
        }
        list_path_fields = {"background_paths", "rir_paths"}
        dict_path_fields = {"negative_feature_data_files"}
        converted = dict(data)
        for field_name in path_fields:
            converted[field_name] = Path(converted.get(field_name, ""))
        for field_name in list_path_fields:
            converted[field_name] = [
                Path(item) for item in converted.get(field_name, [])
            ]
        for field_name in dict_path_fields:
            converted[field_name] = {
                name: Path(path)
                for name, path in converted.get(field_name, {}).items()
            }
        return cls(**converted)
