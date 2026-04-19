"""Tests for cascade-only passive NC trajectories."""

from __future__ import annotations

from random import Random

from cascading_rl.dynamics.cascade import build_initial_state
from cascading_rl.evaluation.passive_trajectory import passive_nc_trajectory
from cascading_rl.graph.generation import make_ba_graph


def test_passive_nc_trajectory_length_and_bounds():
    graph = make_ba_graph(n=25, m=2, seed=42)
    rng = Random(0)
    state = build_initial_state(graph, alpha=0.25, pfail=0.2, rng=rng)
    series = passive_nc_trajectory(state, max_rounds=5)
    assert len(series) == 6
    for x in series:
        assert 0.0 <= x <= 1.0


def test_passive_nc_trajectory_zero_rounds():
    graph = make_ba_graph(n=20, m=2, seed=1)
    rng = Random(1)
    state = build_initial_state(graph, alpha=0.25, pfail=0.2, rng=rng)
    series = passive_nc_trajectory(state, max_rounds=0)
    assert len(series) == 1
