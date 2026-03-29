from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from math import sqrt
from statistics import mean, pstdev

import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import rollout_policy

Policy = Callable[[RecoveryObservation], object]


def estimate_minimum_budget(
    graph: nx.Graph,
    policy: Policy,
    tau: float = 0.8,
    budgets: Iterable[int] = range(1, 11),
    trials: int = 100,
    alpha: float = 0.2,
    pfail: float = 0.1,
    max_rounds: int | None = None,
    env_kwargs: Mapping[str, object] | None = None,
) -> tuple[int | None, dict[int, tuple[float, float]]]:
    """Estimate the smallest budget whose expected final ANC exceeds tau."""
    if trials < 1:
        raise ValueError("trials must be at least 1.")
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1].")

    results: dict[int, tuple[float, float]] = {}
    minimum_budget: int | None = None
    resolved_env_kwargs = dict(env_kwargs or {})

    for budget in budgets:
        anc_values = []
        for seed in range(trials):
            env = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=budget,
                max_rounds=max_rounds,
                seed=seed,
                **resolved_env_kwargs,
            )
            episode = rollout_policy(env, policy, seed=seed, tau=tau)
            anc_values.append(episode.final_anc)

        avg_anc = mean(anc_values)
        stderr = 0.0 if len(anc_values) == 1 else pstdev(anc_values) / sqrt(len(anc_values))
        results[budget] = (avg_anc, stderr)
        if minimum_budget is None and avg_anc >= tau:
            minimum_budget = budget

    return minimum_budget, results
