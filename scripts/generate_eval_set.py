from __future__ import annotations

import sys
from pathlib import Path
from random import Random

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import compute_scaled_budget
from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation.regime import build_policy_factories
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.evaluation.saved_eval_sets import (
    DIAGNOSTIC_POLICY_NAMES,
    EVAL_SPREAD_FILTER_DEGREE_RANDOM,
    regime_label_from_heuristic_rollouts,
    rollout_final_anc_on_instance,
    save_eval_instances,
)

OUTPUT_REL = Path("eval_sets") / "ds_validation.pkl"
NUM_GRAPHS = 30
SEEDS_PER_GRAPH = 5
ALPHA = 0.15
P_FAIL = 0.18
B_REF = 3
N_REF = 40


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def resolve_env_kwargs(config: dict) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    abandon_raw = regime.get("abandonment_anc_threshold")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
        "abandonment_anc_threshold": (
            float(abandon_raw) if abandon_raw is not None else None
        ),
    }


def main() -> None:
    out_path = ROOT / OUTPUT_REL
    if out_path.exists():
        print(f"Eval set already exists at {out_path}; skipping generation.")
        return

    config_path = ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    training = config["training"]
    evaluation = config["evaluation"]
    regime_mapping = config["regime_mapping"]
    m = int(training["graph"]["m"])
    max_rounds = int(training["regime"]["max_rounds"])
    env_kwargs = resolve_env_kwargs(config)

    master_seed = int(training["seed"]) + 50_000
    rng = Random(master_seed)
    graphs_meta: list[tuple[object, int, int]] = []
    for _ in range(NUM_GRAPHS):
        n = rng.randint(30, 50)
        graph_seed = rng.randint(0, 10**9)
        graphs_meta.append((make_ba_graph(n=n, m=m, seed=graph_seed), n, graph_seed))

    base_failure_seed = master_seed + 1_000_000
    generated = 0
    kept: list[dict] = []
    spreads: list[float] = []

    factories = build_policy_factories(base_seed=master_seed)

    for gi, (graph, n, graph_seed) in enumerate(graphs_meta):
        b_scaled = compute_scaled_budget(
            B_REF,
            num_nodes=n,
            reference_n=N_REF,
            enabled=True,
        )
        for s in range(SEEDS_PER_GRAPH):
            generated += 1
            failure_seed = base_failure_seed + gi * 1000 + s
            probe = RecoveryEnv(
                graph,
                alpha=ALPHA,
                pfail=P_FAIL,
                budget=b_scaled,
                max_rounds=max_rounds,
                seed=0,
                **env_kwargs,
            )
            obs = probe.reset(seed=failure_seed)
            if not obs.failed:
                continue

            initial_failures = frozenset(obs.failed)
            pol_degree = factories["degree"](gi, failure_seed)
            pol_random = factories["random"](gi, failure_seed)
            pr_degree = rollout_final_anc_on_instance(
                graph,
                alpha=ALPHA,
                p_fail=P_FAIL,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_degree,
            )
            pr_random = rollout_final_anc_on_instance(
                graph,
                alpha=ALPHA,
                p_fail=P_FAIL,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_random,
            )
            spread = pr_degree - pr_random
            if spread <= EVAL_SPREAD_FILTER_DEGREE_RANDOM:
                continue

            spreads.append(spread)
            regime_label = regime_label_from_heuristic_rollouts(
                graph,
                alpha=ALPHA,
                p_fail=P_FAIL,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                hopeless_threshold=float(regime_mapping["hopeless_threshold"]),
                trivial_threshold=float(regime_mapping["trivial_threshold"]),
                spread_threshold=float(regime_mapping["spread_threshold"]),
                base_seed=master_seed,
                graph_index=gi,
            )
            kept.append(
                {
                    "graph": graph,
                    "initial_failures": initial_failures,
                    "alpha": ALPHA,
                    "p_fail": P_FAIL,
                    "budget": b_scaled,
                    "b_scaled": b_scaled,
                    "b_ref": B_REF,
                    "n_ref": N_REF,
                    "n": n,
                    "graph_seed": graph_seed,
                    "failure_seed": failure_seed,
                    "pr_degree": pr_degree,
                    "pr_random": pr_random,
                    "spread": spread,
                    "regime_label": regime_label,
                    "max_rounds": max_rounds,
                    "m": m,
                }
            )

    save_eval_instances(out_path, kept)
    mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
    min_sp = min(spreads) if spreads else 0.0
    max_sp = max(spreads) if spreads else 0.0
    print(f"Wrote {len(kept)} instances to {out_path}")
    print(
        f"Summary: generated_candidates={generated}, kept={len(kept)}, "
        f"mean_spread={mean_spread:.4f}, min_spread={min_sp:.4f}, max_spread={max_sp:.4f}"
    )
    print(f"Heuristic policies used for regime label: {', '.join(DIAGNOSTIC_POLICY_NAMES)}")


if __name__ == "__main__":
    main()
