"""Project persistence service for WakeLab."""
# Author: Clive Bostock
# Date: 2026-05-10
# Description: Creates and loads wake-word project workspaces.

from __future__ import annotations

import json
import shutil
from pathlib import Path

from orac_wake_lab.models.project import DEFAULT_WORKSPACE_ROOT
from orac_wake_lab.models.project import WakeWordProject
from orac_wake_lab.services import wake_lab_home


PROJECT_SUBDIRS = [
    "config",
    "openwakeword_output",
    "models",
    "logs",
    "export",
    "real_positives",
    "test/activations",
    "test/false_positives",
    "test/false_negatives",
    "test/near_misses",
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


def discover_projects(projects_root: Path | None = None) -> list[WakeWordProject]:
    """Load saved projects from the managed projects directory.

    Args:
        projects_root (Path | None): Directory containing project workspaces.
            Defaults to the managed Wake Lab projects root.

    Returns:
        list[WakeWordProject]: Loadable projects sorted by model name.
    """
    root = (projects_root or wake_lab_home.get_projects_root()).expanduser()
    if not root.exists():
        return []

    projects: list[WakeWordProject] = []
    for project_path in sorted(root.glob("*/project.json")):
        projects.append(load_project(project_path))
    return sorted(projects, key=lambda project: project.model_name)


def load_project(project_path: Path) -> WakeWordProject:
    """Load a project from ``project.json``.

    Args:
        project_path (Path): Path to a project JSON file.

    Returns:
        WakeWordProject: Loaded project.
    """
    data = json.loads(project_path.expanduser().read_text(encoding="utf-8"))
    return WakeWordProject.from_json_dict(data)


def delete_project_workspace(project: WakeWordProject) -> bool:
    """Delete a saved project workspace.

    The workspace must contain a ``project.json`` whose saved
    ``workspace_dir`` resolves to the same directory. This keeps deletion
    scoped to valid WakeLab project folders.

    Args:
        project (WakeWordProject): Project workspace to delete.

    Returns:
        bool: True when a workspace was deleted, False when it did not exist.

    Raises:
        ValueError: If the path is not a valid project workspace.
    """
    workspace_dir = project.workspace_dir.expanduser().resolve(strict=False)
    if not workspace_dir.exists():
        return False
    if not workspace_dir.is_dir():
        raise ValueError(
            f"Project workspace is not a directory: {workspace_dir}"
        )

    project_path = workspace_dir / "project.json"
    if not project_path.exists():
        raise ValueError(
            f"Refusing to delete workspace without project.json: {workspace_dir}"
        )

    saved_project = load_project(project_path)
    saved_workspace = saved_project.workspace_dir.expanduser().resolve(
        strict=False
    )
    if saved_workspace != workspace_dir:
        raise ValueError(
            "Refusing to delete workspace because project.json points to "
            f"{saved_workspace}, not {workspace_dir}"
        )

    protected_dirs = {
        Path.home().resolve(),
        wake_lab_home.get_wake_lab_home().resolve(),
        wake_lab_home.get_projects_root().resolve(),
    }
    if (
        workspace_dir in protected_dirs
        or workspace_dir.parent == workspace_dir
    ):
        raise ValueError(
            f"Refusing to delete protected directory: {workspace_dir}"
        )

    shutil.rmtree(workspace_dir)
    return True
