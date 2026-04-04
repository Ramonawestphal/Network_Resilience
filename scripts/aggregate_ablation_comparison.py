"""Merge per-run ablation JSONs into experiments/ablation/ablation_comparison.json.

The aggregate file is derived (and gitignored) so the repo keeps one source of truth
per run under experiments/ablation/<name>.json. Run after training or when adding
new per-run artifacts; requires every run listed in scripts/run_ablation.ABLATION_RUNS.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = importlib.util.spec_from_file_location("run_ablation", ROOT / "scripts" / "run_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    out_dir = mod.ABLATION_OUTPUT_DIR
    configs: list[dict] = []
    missing: list[str] = []

    for run in mod.ABLATION_RUNS:
        name = run["name"]
        path = out_dir / f"{name}.json"
        if not path.is_file():
            missing.append(name)
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("name") != name:
            raise SystemExit(f"{path}: expected name {name!r}, got {payload.get('name')!r}")
        configs.append(payload)

    if missing:
        sys.stderr.write("Missing per-run JSON files:\n  " + "\n  ".join(missing) + "\n")
        raise SystemExit(1)

    out_path = out_dir / "ablation_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"configs": configs}, f, indent=2)
    print(f"Wrote {len(configs)} configs to {out_path}")


if __name__ == "__main__":
    main()
