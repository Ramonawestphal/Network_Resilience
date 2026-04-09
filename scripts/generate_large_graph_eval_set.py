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
    regime_label_from_heuristic_rollouts,
    rollout_final_nc_on_instance,
    save_eval_instances,
)

ALPHA = 0.15
PRIMARY_PFAIL = 0.18
FALLBACK_PFAIL = 0.20
NUM_GRAPHS = 20
SEEDS_PER_GRAPH = 5
KEEP_FRACTION_WARN = 0.3
# Exclude instances where the degree heuristic already achieves very high recovery (not hard enough).
EVAL_DS_MAX_PR_DEGREE = 0.90

SETS = (
    ("large_graph_medium.pkl", (100, 150), 60_000),
    ("large_graph_large.pkl", (300, 500), 70_000),
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def resolve_env_kwargs(config: dict) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    abandon_raw = regime.get("abandonment_nc_threshold")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
        "abandonment_nc_threshold": (
            float(abandon_raw) if abandon_raw is not None else None
        ),
    }


def build_filtered_instances(
    *,
    n_range: tuple[int, int],
    b_ref: int,
    n_ref: int,
    scale_budget: bool,
    m: int,
    max_rounds: int,
    env_kwargs: dict[str, object],
    regime_mapping: dict,
    master_seed: int,
    p_fail: float,
) -> tuple[list[dict], int, list[float]]:
    rng = Random(master_seed)
    spread_threshold = float(regime_mapping["spread_threshold"])
    graphs_meta: list[tuple[object, int, int]] = []
    for _ in range(NUM_GRAPHS):
        n = rng.randint(n_range[0], n_range[1])
        graph_seed = rng.randint(0, 10**9)
        graphs_meta.append((make_ba_graph(n=n, m=m, seed=graph_seed), n, graph_seed))

    base_failure_seed = master_seed + 1_000_000
    generated = 0
    kept: list[dict] = []
    spreads: list[float] = []
    factories = build_policy_factories(base_seed=master_seed)

    for gi, (graph, n, graph_seed) in enumerate(graphs_meta):
        b_scaled = compute_scaled_budget(
            b_ref,
            num_nodes=n,
            reference_n=n_ref,
            enabled=scale_budget,
        )
        for s in range(SEEDS_PER_GRAPH):
            generated += 1
            failure_seed = base_failure_seed + gi * 1000 + s
            probe = RecoveryEnv(
                graph,
                alpha=ALPHA,
                pfail=p_fail,
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
            pr_degree = rollout_final_nc_on_instance(
                graph,
                alpha=ALPHA,
                p_fail=p_fail,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_degree,
            )
            pr_random = rollout_final_nc_on_instance(
                graph,
                alpha=ALPHA,
                p_fail=p_fail,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_random,
            )
            spread = pr_degree - pr_random
            is_ds = (spread > spread_threshold) and (pr_degree < EVAL_DS_MAX_PR_DEGREE)
            if not is_ds:
                continue

            spreads.append(spread)
            regime_label = regime_label_from_heuristic_rollouts(
                graph,
                alpha=ALPHA,
                p_fail=p_fail,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                hopeless_threshold=float(regime_mapping["hopeless_threshold"]),
                trivial_threshold=float(regime_mapping["trivial_threshold"]),
                spread_threshold=spread_threshold,
                base_seed=master_seed,
                graph_index=gi,
            )
            kept.append(
                {
                    "graph": graph,
                    "initial_failures": initial_failures,
                    "alpha": ALPHA,
                    "p_fail": p_fail,
                    "budget": b_scaled,
                    "b_scaled": b_scaled,
                    "b_ref": b_ref,
                    "n_ref": n_ref,
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

    return kept, generated, spreads


def run_one_set(
    filename: str,
    n_range: tuple[int, int],
    seed_offset: int,
    config: dict,
) -> None:
    out_path = ROOT / "eval_sets" / filename
    if out_path.exists():
        print(f"Eval set already exists at {out_path}; skipping.")
        return

    training = config["training"]
    evaluation = config["evaluation"]
    budget_scaling = config["budget_scaling"]
    regime_mapping = config["regime_mapping"]
    m = int(training["graph"]["m"])
    b_ref = int(training["regime"]["budget"])
    n_ref = int(budget_scaling["reference_n"])
    scale_budget = bool(budget_scaling.get("enabled", False))
    max_rounds = int(training["regime"]["max_rounds"])
    env_kwargs = resolve_env_kwargs(config)
    master_seed = int(training["seed"]) + seed_offset

    kept, generated, spreads = build_filtered_instances(
        n_range=n_range,
        b_ref=b_ref,
        n_ref=n_ref,
        scale_budget=scale_budget,
        m=m,
        max_rounds=max_rounds,
        env_kwargs=env_kwargs,
        regime_mapping=regime_mapping,
        master_seed=master_seed,
        p_fail=PRIMARY_PFAIL,
    )
    n_kept_primary = len(kept)
    frac = n_kept_primary / generated if generated else 0.0
    used_pfail = PRIMARY_PFAIL
    if frac < KEEP_FRACTION_WARN:
        print(
            f"WARNING: {filename}: {n_kept_primary}/{generated} passed DS filter "
            f"({frac:.1%}) at p_fail={PRIMARY_PFAIL}; retrying with p_fail={FALLBACK_PFAIL}."
        )
        kept, generated, spreads = build_filtered_instances(
            n_range=n_range,
            b_ref=b_ref,
            n_ref=n_ref,
            scale_budget=scale_budget,
            m=m,
            max_rounds=max_rounds,
            env_kwargs=env_kwargs,
            regime_mapping=regime_mapping,
            master_seed=master_seed + 333_333,
            p_fail=FALLBACK_PFAIL,
        )
        used_pfail = FALLBACK_PFAIL
        n_kept_fb = len(kept)
        frac = n_kept_fb / generated if generated else 0.0
        print(
            f"After fallback at p_fail={FALLBACK_PFAIL}: {n_kept_fb}/{generated} passed DS filter "
            f"({frac:.1%})."
        )

    save_eval_instances(out_path, kept)
    mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
    min_sp = min(spreads) if spreads else 0.0
    max_sp = max(spreads) if spreads else 0.0
    print(f"Wrote {len(kept)} instances to {out_path} (p_fail={used_pfail})")
    print(
        f"Summary [{filename}]: generated_candidates={generated}, kept={len(kept)}, "
        f"mean_spread={mean_spread:.4f}, min_spread={min_sp:.4f}, max_spread={max_sp:.4f}"
    )
    print(f"Heuristic policies used for regime label: {', '.join(DIAGNOSTIC_POLICY_NAMES)}")


def main() -> None:
    (ROOT / "eval_sets").mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "config" / "default.yaml")
    for filename, n_range, seed_off in SETS:
        run_one_set(filename, n_range, seed_off, config)


if __name__ == "__main__":
    main()
