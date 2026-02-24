from __future__ import annotations

from pathlib import Path
import subprocess


def detect_llambo_repo(root: Path) -> tuple[Path, str | None]:
    repo_path = root / "external" / "LLAMBO"
    if not repo_path.exists():
        return repo_path, None

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    return repo_path, commit
