from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from cascading_rl.evaluation.saved_eval_sets import (
    load_eval_instances,
    recovery_env_from_instance,
    save_eval_instances,
)


def test_save_and_load_eval_instances_roundtrip(tmp_path: Path):
    graph = nx.path_graph(4)
    instances = [
        {
            "graph": graph,
            "initial_failures": frozenset({2}),
            "alpha": 0.15,
            "p_fail": 0.18,
            "budget": 3,
            "graph_seed": 1,
            "failure_seed": 2,
            "pr_degree": 0.5,
            "pr_random": 0.3,
            "spread": 0.2,
            "regime_label": "decision-sensitive",
            "max_rounds": 5,
            "m": 2,
        }
    ]
    path = tmp_path / "set.pkl"
    save_eval_instances(path, instances)
    loaded = load_eval_instances(path)
    assert len(loaded) == 1
    assert loaded[0]["budget"] == 3
    assert loaded[0]["graph"].number_of_nodes() == 4
    assert loaded[0]["initial_failures"] == frozenset({2})
    assert isinstance(loaded[0]["graph"], nx.Graph)


def test_recovery_env_from_instance_prefers_b_scaled_when_present():
    env_kwargs = {
        "capacity_noise": 0.0,
        "failure_bias": "uniform",
        "action_space": "failed",
        "obs_hops": None,
    }
    inst = {
        "graph": nx.path_graph(12),
        "alpha": 0.2,
        "p_fail": 0.1,
        "budget": 3,
        "b_scaled": 11,
        "max_rounds": 5,
    }
    env = recovery_env_from_instance(inst, env_kwargs=env_kwargs)
    assert env.budget == 11


def test_pickle_protocol_for_eval_set(tmp_path: Path):
    path = tmp_path / "raw.pkl"
    with path.open("wb") as f:
        pickle.dump([{"k": 1}], f, protocol=4)
    assert load_eval_instances(path) == [{"k": 1}]
