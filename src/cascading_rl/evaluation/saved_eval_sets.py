"""Helpers for fixed pickle eval sets (Phase 3) and heuristic spread / regime labels."""

from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import (
    EpisodeResult,
    PolicyEvaluationSummary,
    rollout_policy,
    summarize_episode_results,
)
from cascading_rl.evaluation.regime import build_policy_factories, compute_regime_diagnostics

# Filter: degree vs random final_anc spread (not regime_mapping.spread_threshold).
EVAL_SPREAD_FILTER_DEGREE_RANDOM = 0.15

DIAGNOSTIC_POLICY_NAMES: tuple[str, ...] = (
    "degree",
    "random",
    "risk",
    "greedy",
    "betweenness",
)


def recovery_env_from_instance(
    inst: Mapping[str, Any],
    *,
    env_kwargs: Mapping[str, object],
) -> RecoveryEnv:
    """Build env for one saved instance.

    Per-instance budget is ``b_scaled`` when present (large-graph sets), otherwise
    ``budget`` (e.g. fixed B=3 for ``ds_validation.pkl``). Callers evaluating
    official large-graph pickles should validate ``b_scaled`` before calling.
    """
    budget = int(inst["b_scaled"]) if "b_scaled" in inst else int(inst["budget"])
    return RecoveryEnv(
        inst["graph"],
        alpha=float(inst["alpha"]),
        pfail=float(inst["p_fail"]),
        budget=budget,
        max_rounds=int(inst["max_rounds"]),
        seed=0,
        **dict(env_kwargs),
    )


def rollout_final_anc_on_instance(
    graph: nx.Graph,
    *,
    alpha: float,
    p_fail: float,
    budget: int,
    max_rounds: int,
    failure_seed: int,
    env_kwargs: Mapping[str, object],
    policy: Callable[[RecoveryObservation], Any],
    tau: float,
) -> float:
    env = RecoveryEnv(
        graph,
        alpha=alpha,
        pfail=p_fail,
        budget=budget,
        max_rounds=max_rounds,
        seed=0,
        **dict(env_kwargs),
    )
    return rollout_policy(env, policy, seed=failure_seed, tau=tau).final_anc


def regime_label_from_heuristic_rollouts(
    graph: nx.Graph,
    *,
    alpha: float,
    p_fail: float,
    budget: int,
    max_rounds: int,
    failure_seed: int,
    env_kwargs: Mapping[str, object],
    tau: float,
    hopeless_threshold: float,
    trivial_threshold: float,
    spread_threshold: float,
    base_seed: int = 0,
    graph_index: int = 0,
) -> str:
    factories = build_policy_factories(base_seed=base_seed)
    summaries: dict[str, Any] = {}
    for name in DIAGNOSTIC_POLICY_NAMES:
        policy = factories[name](graph_index, failure_seed)
        env = RecoveryEnv(
            graph,
            alpha=alpha,
            pfail=p_fail,
            budget=budget,
            max_rounds=max_rounds,
            seed=0,
            **dict(env_kwargs),
        )
        result = rollout_policy(env, policy, seed=failure_seed, tau=tau)
        summaries[name] = summarize_episode_results([result])
    diagnostics = compute_regime_diagnostics(
        summaries,
        hopeless_threshold=hopeless_threshold,
        trivial_threshold=trivial_threshold,
        spread_threshold=spread_threshold,
        budget_sensitivity=None,
    )
    return diagnostics.regime_label


def load_eval_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Eval set {path} must contain a list of instance dicts.")
    return data


def save_eval_instances(path: Path, instances: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(list(instances), f, protocol=4)


def evaluate_policies_on_saved_instances(
    instances: Sequence[Mapping[str, Any]],
    policy_factories: Mapping[str, object],
    *,
    env_kwargs: Mapping[str, object],
    tau: float,
    policy_names: Sequence[str],
) -> tuple[
    dict[str, PolicyEvaluationSummary],
    dict[str, dict[str, PolicyEvaluationSummary]],
]:
    """Aggregate rollouts over all instances; second return groups by instance regime_label."""
    by_policy: dict[str, list[EpisodeResult]] = {name: [] for name in policy_names}
    by_regime: dict[str, dict[str, list[EpisodeResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx, inst in enumerate(instances):
        label = str(inst.get("regime_label", "unknown"))
        seed_i = int(inst["failure_seed"])
        for name in policy_names:
            env = recovery_env_from_instance(inst, env_kwargs=env_kwargs)
            policy = policy_factories[name](idx, seed_i)
            result = rollout_policy(env, policy, seed=seed_i, tau=tau)
            by_policy[name].append(result)
            by_regime[label][name].append(result)
    overall = {n: summarize_episode_results(rs) for n, rs in by_policy.items() if rs}
    per_bucket = {
        lbl: {n: summarize_episode_results(rs) for n, rs in pmap.items() if rs}
        for lbl, pmap in by_regime.items()
    }
    return overall, per_bucket


def mean_final_anc_from_summaries(
    summaries: dict[str, PolicyEvaluationSummary],
    policies: Sequence[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in policies:
        if name in summaries:
            out[name] = summaries[name].final_anc.mean
    return out
