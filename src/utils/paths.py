from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def default_third_party_data_dir() -> Path:
    return PROJECT_ROOT / "src" / "data_mgmt" / "data_tools" / "third_party_device_data"