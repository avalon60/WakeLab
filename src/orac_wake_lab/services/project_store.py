"""Project persistence service for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Creates and loads wake-word project workspaces.

from __future__ import annotations

import json
from pathlib import Path

from orac_wake_lab.models.project import DEFAULT_WORKSPACE_ROOT
from orac_wake_lab.models.project import WakeWordProject


PROJECT_SUBDIRS = [
    "config",
    "openwakeword_output",
    "models",
    "logs",
    "export",
    "test/activations",
    "test/false_positives",
    "test/false_negatives",
]


def project_dir_for_model(
    model_name: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> Path:
    """Return the default project directory for a model name.

    Args:
        model_name (str): Valid model name.
        workspace_root (Path): Root directory for projects.

    Returns:
        Path: Project directory path.
    """
    return workspace_root.expanduser() / model_name


def create_project_workspace(project: WakeWordProject) -> WakeWordProject:
    """Create a project workspace and persist ``project.json``.

    Args:
        project (WakeWordProject): Project to create.

    Returns:
        WakeWordProject: The same project instance.
    """
    project.workspace_dir.mkdir(parents=True, exist_ok=True)
    for subdir in PROJECT_SUBDIRS:
        (project.workspace_dir / subdir).mkdir(parents=True, exist_ok=True)
    save_project(project)
    return project


def save_project(project: WakeWordProject) -> Path:
    """Persist project metadata to ``project.json``.

    Args:
        project (WakeWordProject): Project to save.

    Returns:
        Path: Written project metadata path.
    """
    project_path = project.workspace_dir / "project.json"
    project_path.write_text(
        json.dumps(project.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return project_path


def load_project(project_path: Path) -> WakeWordProject:
    """Load a project from ``project.json``.

    Args:
        project_path (Path): Path to a project JSON file.

    Returns:
        WakeWordProject: Loaded project.
    """
    data = json.loads(project_path.expanduser().read_text(encoding="utf-8"))
    return WakeWordProject.from_json_dict(data)
