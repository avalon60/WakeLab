"""Bootstrap helpers for openWakeWord training assets."""
# Author: Clive Bostock
# Date: 2026-05-13
# Description: Ensures the upstream openWakeWord ONNX assets are available.

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


MELSPECTROGRAM_URL = (
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
    "melspectrogram.onnx"
)
EMBEDDING_MODEL_URL = (
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
    "embedding_model.onnx"
)


def ensure_openwakeword_training_assets(repo_path: Path) -> list[Path]:
    """Ensure the melspectrogram and embedding ONNX assets exist.

    Args:
        repo_path (Path): Local openWakeWord repository root.

    Returns:
        list[Path]: Assets that were downloaded during this call.

    Raises:
        RuntimeError: If a download fails.
    """
    models_dir = repo_path / "openwakeword" / "resources" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    downloads = [
        (models_dir / "melspectrogram.onnx", MELSPECTROGRAM_URL),
        (models_dir / "embedding_model.onnx", EMBEDDING_MODEL_URL),
    ]
    downloaded: list[Path] = []
    for asset_path, url in downloads:
        if asset_path.exists() and asset_path.stat().st_size > 0:
            continue
        _download_file(url, asset_path)
        downloaded.append(asset_path)
    return downloaded


def openwakeword_training_assets_status(repo_path: Path) -> tuple[bool, list[Path]]:
    """Return whether the training assets are present.

    Args:
        repo_path (Path): Local openWakeWord repository root.

    Returns:
        tuple[bool, list[Path]]: Presence flag and missing asset paths.
    """
    models_dir = repo_path / "openwakeword" / "resources" / "models"
    expected = [
        models_dir / "melspectrogram.onnx",
        models_dir / "embedding_model.onnx",
    ]
    missing = [path for path in expected if not path.exists() or path.stat().st_size <= 0]
    return (not missing, missing)


def _download_file(url: str, destination: Path) -> None:
    """Download a file to disk with a temporary staging path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urlopen(url) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temp_path.replace(destination)
    except URLError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            f"Unable to download {destination.name} from {url}: {exc}"
        ) from exc
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
