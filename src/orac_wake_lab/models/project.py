"""Project data model for WakeLab."""
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
    """Represent a WakeLab wake-word project.

    Args:
        wake_phrase (str): Wake phrase to train.
        model_name (str): Filesystem/model-safe model name.
        workspace_dir (Path): Project workspace directory.
        openwakeword_repo (Path): Local openWakeWord repository path.
        orac_repo (Path): Optional runtime target root path.
        piper_sample_generator_path (Path): Piper sample generator path.
        piper_voice_model_path (Path): Piper voice or generator model path.
        use_training_phrase_parts (bool): Whether to synthesize positives
          from the split phrase-parts fields instead of the simple phrase.
        training_pronunciation_phrase (str): Optional positive-sample
          pronunciation phrase used only for TTS generation.
        training_phrase_parts (list[str]): Optional positive-sample phrase
          parts to generate separately before concatenation.
        inter_part_silence_min_ms (int): Minimum generated silence between
          phrase parts, in milliseconds.
        inter_part_silence_max_ms (int): Maximum generated silence between
          phrase parts, in milliseconds.
        enable_real_positives (bool): Whether imported user recordings should
          be added to positive training data.
        real_positive_clips_dir (Path): Project-managed real positive clips
          directory.
        real_positive_min_count (int): Minimum preferred real-positive clip
          count before training.
        real_positive_target_percent (int): Target percentage of staged
          positive training clips that should come from real recordings.
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
    piper_voice_model_path: Path = Path("")
    use_training_phrase_parts: bool = False
    training_pronunciation_phrase: str = ""
    training_phrase_parts: list[str] | None = None
    inter_part_silence_min_ms: int = 80
    inter_part_silence_max_ms: int = 250
    enable_real_positives: bool = True
    real_positive_clips_dir: Path = Path("")
    real_positive_min_count: int = 20
    real_positive_target_percent: int = 30
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
        self.piper_voice_model_path = self.piper_voice_model_path.expanduser()
        if isinstance(self.use_training_phrase_parts, str):
            self.use_training_phrase_parts = (
                self.use_training_phrase_parts.strip().lower()
                not in {"", "0", "false", "no", "off"}
            )
        else:
            self.use_training_phrase_parts = bool(
                self.use_training_phrase_parts
            )
        self.training_pronunciation_phrase = (
            self.training_pronunciation_phrase.strip()
        )
        self.training_phrase_parts = [
            part.strip() for part in (self.training_phrase_parts or [])
        ]
        self.inter_part_silence_min_ms = int(self.inter_part_silence_min_ms)
        self.inter_part_silence_max_ms = int(self.inter_part_silence_max_ms)
        if isinstance(self.enable_real_positives, str):
            self.enable_real_positives = (
                self.enable_real_positives.strip().lower()
                not in {"", "0", "false", "no", "off"}
            )
        else:
            self.enable_real_positives = bool(self.enable_real_positives)
        self.real_positive_clips_dir = self.real_positives_dir
        self.real_positive_min_count = int(self.real_positive_min_count)
        self.real_positive_target_percent = min(
            100,
            max(10, int(self.real_positive_target_percent)),
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
    def clip_dir(self) -> Path:
        """Return the project clip workspace directory."""
        return self.workspace_dir / "clips"

    @property
    def generated_clips_dir(self) -> Path:
        """Return the generated clip output directory."""
        return self.clip_dir / "generated"

    @property
    def augmented_clips_dir(self) -> Path:
        """Return the augmented clip output directory."""
        return self.clip_dir / "augmented"

    @property
    def real_positives_dir(self) -> Path:
        """Return the managed real positive clips directory."""
        return self.workspace_dir / "real_positives"

    @property
    def near_miss_clips_dir(self) -> Path:
        """Return the managed near-miss evaluation clips directory."""
        return self.workspace_dir / "test" / "near_misses"

    @property
    def training_config_path(self) -> Path:
        """Return the generated training config path."""
        return self.config_dir / "training.yaml"

    @property
    def orac_candidate_config_path(self) -> Path:
        """Return the legacy candidate config snippet path."""
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
            "piper_voice_model_path",
            "real_positive_clips_dir",
            "false_positive_validation_data_path",
        }
        list_path_fields = {"background_paths", "rir_paths"}
        dict_path_fields = {"negative_feature_data_files"}
        converted = dict(data)
        if "real_positive_repeat_count" in converted:
            converted.pop("real_positive_repeat_count")
        if "use_training_phrase_parts" not in converted:
            converted["use_training_phrase_parts"] = bool(
                converted.get("training_phrase_parts")
            )
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
