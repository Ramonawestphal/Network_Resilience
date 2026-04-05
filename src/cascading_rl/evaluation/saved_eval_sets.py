"""Helpers for fixed eval sets (JSON, YAML, or pickle with a ``.pkl`` suffix) and heuristic spread."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import yaml  # type: ignore[import-untyped]
from networkx.readwrite import json_graph

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import (
    EpisodeResult,
    PolicyEvaluationSummary,
    final_anc_failure_threshold_for_reporting,
    rollout_policy,
    summarize_episode_results,
)
from cascading_rl.evaluation.regime import build_policy_factories, compute_regime_diagnostics

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
    ``budget`` (e.g. fixed B=3 for ``ds_validation.json``).
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
    return rollout_policy(env, policy, seed=failure_seed).final_anc


def regime_label_from_heuristic_rollouts(
    graph: nx.Graph,
    *,
    alpha: float,
    p_fail: float,
    budget: int,
    max_rounds: int,
    failure_seed: int,
    env_kwargs: Mapping[str, object],
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
        result = rollout_policy(env, policy, seed=failure_seed)
        summaries[name] = summarize_episode_results(
            [result],
            final_anc_failure_threshold=thr,
        )
    diagnostics = compute_regime_diagnostics(
        summaries,
        hopeless_threshold=hopeless_threshold,
        trivial_threshold=trivial_threshold,
        spread_threshold=spread_threshold,
        budget_sensitivity=None,
    )
    return diagnostics.regime_label


def _instance_to_serializable(inst: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in inst.items():
        if key == "graph":
            if not isinstance(value, nx.Graph):
                raise TypeError(
                    f"instance['graph'] must be a networkx.Graph, got {type(value).__name__}"
                )
            out[key] = json_graph.node_link_data(value)
        elif key == "initial_failures":
            if isinstance(value, frozenset):
                out[key] = sorted(value)
            elif isinstance(value, (list, tuple)):
                out[key] = list(value)
            else:
                raise TypeError(
                    f"instance['initial_failures'] must be frozenset, list, or tuple, "
                    f"got {type(value).__name__}"
                )
        else:
            out[key] = value
    return out


def _instance_from_decoded(item: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(item)
    graph_raw = out.get("graph")
    if isinstance(graph_raw, dict):
        if "nodes" not in graph_raw:
            raise ValueError(
                "Each instance dict must encode 'graph' as networkx node_link_data "
                "(requires a 'nodes' list)."
            )
        if "links" not in graph_raw and "edges" not in graph_raw:
            raise ValueError(
                "Each instance dict must encode 'graph' as networkx node_link_data "
                "(requires 'edges' or 'links')."
            )
        out["graph"] = json_graph.node_link_graph(graph_raw)
    elif isinstance(graph_raw, nx.Graph):
        pass
    elif graph_raw is not None:
        raise TypeError(
            f"instance['graph'] must be node_link_data dict or Graph, got {type(graph_raw).__name__}"
        )
    if "initial_failures" in out:
        ib = out["initial_failures"]
        if isinstance(ib, list):
            out["initial_failures"] = frozenset(ib)
        elif isinstance(ib, frozenset):
            pass
        else:
            raise TypeError(
                f"instance['initial_failures'] must be list or frozenset, got {type(ib).__name__}"
            )
    return out


def _load_eval_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    # Be tolerant of mislabeled eval-set files: some existing artifacts use a
    # `.json` suffix even though the payload is a pickle stream.
    if suffix == ".pkl" or raw.startswith(b"\x80"):
        return pickle.loads(raw)
    text = raw.decode("utf-8")
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if suffix == ".json" or suffix == "":
        return json.loads(text)
    raise ValueError(
        f"Eval set {path}: expected extension .json, .yaml, .yml, or .pkl (got {path.suffix!r})."
    )


def load_eval_instances(path: Path) -> list[dict[str, Any]]:
    data = _load_eval_payload(path)
    if not isinstance(data, list):
        raise ValueError(f"Eval set {path} must contain a list of instance dicts.")
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Eval set {path}: each entry must be a mapping, got {type(item).__name__}"
            )
        out.append(_instance_from_decoded(item))
    return out


def save_eval_instances(path: Path, instances: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("wb") as f:
            pickle.dump(list(instances), f, protocol=4)
        return
    payload = [_instance_to_serializable(inst) for inst in instances]
    if suffix in {".json", ""}:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    if suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return
    raise ValueError(
        f"Eval set {path}: cannot save as {path.suffix!r}; use .json, .yaml, .yml, or .pkl."
    )


def evaluate_policies_on_saved_instances(
    instances: Sequence[Mapping[str, Any]],
    policy_factories: Mapping[str, object],
    *,
    env_kwargs: Mapping[str, object],
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
            result = rollout_policy(env, policy, seed=seed_i)
            by_policy[name].append(result)
            by_regime[label][name].append(result)
    thr = final_anc_failure_threshold_for_reporting(env_kwargs)
    overall = {
        n: summarize_episode_results(rs, final_anc_failure_threshold=thr)
        for n, rs in by_policy.items()
        if rs
    }
    per_bucket = {
        lbl: {
            n: summarize_episode_results(rs, final_anc_failure_threshold=thr)
            for n, rs in pmap.items()
            if rs
        }
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
