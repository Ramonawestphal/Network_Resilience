from __future__ import annotations

import importlib.util
from pathlib import Path

from cascading_rl.models import FEATURE_NAMES, GLOBAL_FEATURE_NAMES


def _load_run_ablation_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_ablation_runs_includes_virtual_and_non_virtual_leave_one_out_variants():
    run_ablation = _load_run_ablation_module()

    runs = run_ablation.build_ablation_runs()
    run_names = {run["name"] for run in runs}
    run_by_name = {run["name"]: run for run in runs}

    for feature_name in GLOBAL_FEATURE_NAMES:
        for suffix, use_virtual_node in (("", False), ("_virtual", True)):
            run_name = f"drop_global_{feature_name}{suffix}"
            assert run_name in run_names
            assert run_by_name[run_name]["use_virtual_node"] is use_virtual_node
            assert feature_name not in run_by_name[run_name]["active_global_features"]

    for feature_name in FEATURE_NAMES:
        for suffix, use_virtual_node in (("", False), ("_virtual", True)):
            run_name = f"drop_node_{feature_name}{suffix}"
            assert run_name in run_names
            assert run_by_name[run_name]["use_virtual_node"] is use_virtual_node
            assert feature_name not in run_by_name[run_name]["active_node_features"]
