from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_config_hash(config_path: Path | None) -> str | None:
    if config_path is None or not config_path.exists():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _resolve_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
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
        "script": str(script_path),
        "argv": list(argv),
        "config_path": str(config_path) if config_path is not None else None,
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
