from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repository root (parent of src/) for git metadata when installed from source tree.
REPO_ROOT = Path(__file__).resolve().parents[2]


def portable_artifact_path(path: str | Path) -> str:
    """Return a POSIX path relative to ``REPO_ROOT`` for portable JSON artifacts.

    If the path is not under the repository root, returns the resolved absolute path
    as a normalized POSIX string (forward slashes on all platforms).
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def portable_repo_relative_path(path: str | Path) -> str:
    """Normalize *path* for configs and summaries: repo-relative POSIX when under ``REPO_ROOT``.

    Relative *path* is resolved against ``REPO_ROOT`` (not the process CWD). If the result
    is still outside the repo (e.g. user file), return absolute POSIX.
    """
    p = Path(path)
    if not p.is_absolute():
        candidate = (REPO_ROOT / p).resolve()
    else:
        candidate = p.resolve()
    try:
        return candidate.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def _portable_argv(argv: list[str]) -> list[str]:
    """Best-effort: rewrite argv entries that are existing paths under the repo as relative POSIX."""
    out: list[str] = []
    for arg in argv:
        try:
            candidate = Path(arg)
            if candidate.exists():
                out.append(portable_artifact_path(candidate))
            else:
                out.append(arg)
        except OSError:
            out.append(arg)
    return out


def _read_config_hash(config_path: Path | None) -> str | None:
    if config_path is None or not config_path.exists():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _resolve_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _torch_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return str(torch.__version__)


def build_run_metadata(
    *,
    script_path: Path,
    argv: list[str],
    config_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": portable_artifact_path(script_path),
        "argv": _portable_argv(list(argv)),
        "config_path": portable_artifact_path(config_path) if config_path is not None else None,
        "config_sha256": _read_config_hash(config_path),
        "git_commit": _resolve_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": _torch_version(),
        **(extra or {}),
    }


def write_run_metadata(
    output_path: Path,
    *,
    script_path: Path,
    argv: list[str],
    config_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_run_metadata(
        script_path=script_path,
        argv=argv,
        config_path=config_path,
        extra=extra,
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
