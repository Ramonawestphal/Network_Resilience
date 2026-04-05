from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from math import sqrt
from statistics import mean, pstdev

import networkx as nx

from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import rollout_policy

Policy = Callable[[RecoveryObservation], object]


def estimate_minimum_budget(
    graph: nx.Graph,
    policy: Policy,
    target_solved_fraction: float = 0.8,
    budgets: Iterable[int] = range(1, 11),
    trials: int = 100,
    alpha: float = 0.2,
    pfail: float = 0.1,
    max_rounds: int | None = None,
    env_kwargs: Mapping[str, object] | None = None,
    scale_budget: bool = False,
    scale_max_rounds: bool = False,
    reference_n: int = 40,
) -> tuple[int | None, dict[int, tuple[float, float]]]:
    """Estimate the smallest budget whose mean fully-restored rate reaches ``target_solved_fraction``."""
    if trials < 1:
        raise ValueError("trials must be at least 1.")
    if not 0.0 <= target_solved_fraction <= 1.0:
        raise ValueError("target_solved_fraction must lie in [0, 1].")

    results: dict[int, tuple[float, float]] = {}
    minimum_budget: int | None = None
    resolved_env_kwargs = dict(env_kwargs or {})

    resolved_max_rounds = (
        compute_scaled_max_rounds(
            max_rounds,
            num_nodes=graph.number_of_nodes(),
            reference_n=reference_n,
            enabled=scale_max_rounds,
        )
        if max_rounds is not None
        else None
    )
    for budget in budgets:
        anc_values = []
        resolved_budget = compute_scaled_budget(
            budget,
            num_nodes=graph.number_of_nodes(),
            reference_n=reference_n,
            enabled=scale_budget,
        )
        for seed in range(trials):
            env = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=resolved_budget,
                max_rounds=resolved_max_rounds,
                seed=seed,
                **resolved_env_kwargs,
            )
            episode = rollout_policy(env, policy, seed=seed)
            anc_values.append(1.0 if episode.remaining_failed_nodes == 0 else 0.0)

        avg_solved = mean(anc_values)
        stderr = 0.0 if len(anc_values) == 1 else pstdev(anc_values) / sqrt(len(anc_values))
        results[budget] = (avg_solved, stderr)
        if minimum_budget is None and avg_solved >= target_solved_fraction:
            minimum_budget = budget

    return minimum_budget, results
