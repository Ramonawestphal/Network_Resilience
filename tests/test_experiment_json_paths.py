"""Guardrails: tracked experiment JSON must not store machine-local absolute paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cascading_rl.reproducibility import REPO_ROOT

_EXPERIMENTS = REPO_ROOT / "experiments"
_WINDOWS_ABS = re.compile(r"[A-Za-z]:\\")
_UNIX_USERS = re.compile(r"^/Users/|^/home/[^/]+/", re.IGNORECASE)


def _iter_strings(obj: object):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


def test_committed_experiment_json_has_no_machine_local_paths() -> None:
    if not _EXPERIMENTS.is_dir():
        pytest.skip("no experiments/ directory")

    failures: list[str] = []
    for path in sorted(_EXPERIMENTS.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            failures.append(f"{path.relative_to(REPO_ROOT)}: invalid JSON ({err})")
            continue
        for string in _iter_strings(data):
            if _WINDOWS_ABS.search(string) or _UNIX_USERS.search(string):
                snippet = string if len(string) <= 160 else string[:157] + "..."
                failures.append(f"{path.relative_to(REPO_ROOT)}: {snippet!r}")

    assert not failures, "machine-local paths in experiment JSON:\n" + "\n".join(failures)
