from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import generate_large_graph_eval_set


def test_build_filtered_instances_respects_disabled_budget_scaling(monkeypatch):
    monkeypatch.setattr(generate_large_graph_eval_set, "NUM_GRAPHS", 1)
    monkeypatch.setattr(generate_large_graph_eval_set, "SEEDS_PER_GRAPH", 1)

    enabled_values: list[bool] = []

    def fake_compute_scaled_budget(
        reference_budget: int,
        *,
        num_nodes: int,
        reference_n: int,
        enabled: bool,
    ) -> int:
        enabled_values.append(enabled)
        return reference_budget

    monkeypatch.setattr(
        generate_large_graph_eval_set, "compute_scaled_budget", fake_compute_scaled_budget
    )
    monkeypatch.setattr(
        generate_large_graph_eval_set, "make_ba_graph", lambda **_: object()
    )
    monkeypatch.setattr(
        generate_large_graph_eval_set,
        "build_policy_factories",
        lambda base_seed: {
            "degree": lambda gi, failure_seed: "degree",
            "random": lambda gi, failure_seed: "random",
        },
    )

    class FakeRecoveryEnv:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed: int):
            return SimpleNamespace(failed={1})

    monkeypatch.setattr(generate_large_graph_eval_set, "RecoveryEnv", FakeRecoveryEnv)
    monkeypatch.setattr(
        generate_large_graph_eval_set,
        "rollout_final_nc_on_instance",
        lambda *args, policy, **kwargs: 0.4 if policy == "degree" else 0.1,
    )
    monkeypatch.setattr(
        generate_large_graph_eval_set,
        "regime_label_from_heuristic_rollouts",
        lambda *args, **kwargs: "decision-sensitive",
    )

    kept, generated, spreads = generate_large_graph_eval_set.build_filtered_instances(
        n_range=(100, 100),
        b_ref=3,
        n_ref=40,
        scale_budget=False,
        m=2,
        max_rounds=20,
        env_kwargs={},
        regime_mapping={
            "spread_threshold": 0.05,
            "hopeless_threshold": 0.25,
            "trivial_threshold": 0.75,
        },
        master_seed=123,
        p_fail=0.18,
    )

    assert enabled_values == [False]
    assert generated == 1
    assert len(kept) == 1
    assert spreads == pytest.approx([0.3])


def test_run_one_set_passes_disabled_budget_scaling_to_primary_and_fallback(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(generate_large_graph_eval_set, "ROOT", tmp_path)

    scale_budget_calls: list[bool] = []

    def fake_build_filtered_instances(**kwargs):
        scale_budget_calls.append(kwargs["scale_budget"])
        if len(scale_budget_calls) == 1:
            return [], 10, []
        return [{"regime_label": "decision-sensitive"}], 10, [0.2]

    monkeypatch.setattr(
        generate_large_graph_eval_set,
        "build_filtered_instances",
        fake_build_filtered_instances,
    )
    monkeypatch.setattr(
        generate_large_graph_eval_set, "save_eval_instances", lambda path, kept: None
    )

    config = {
        "training": {
            "seed": 7,
            "graph": {"m": 2},
            "regime": {
                "budget": 3,
                "max_rounds": 20,
                "capacity_noise": 0.0,
                "failure_bias": "uniform",
                "action_space": "failed",
                "obs_hops": None,
            },
        },
        "evaluation": {},
        "budget_scaling": {"enabled": False, "reference_n": 40},
        "regime_mapping": {
            "spread_threshold": 0.05,
            "hopeless_threshold": 0.25,
            "trivial_threshold": 0.75,
        },
    }

    (tmp_path / "eval_sets").mkdir()
    generate_large_graph_eval_set.run_one_set(
        "large_graph_medium.pkl",
        (100, 150),
        60_000,
        config,
    )

    assert scale_budget_calls == [False, False]
