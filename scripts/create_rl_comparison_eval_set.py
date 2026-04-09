"""Generate a fixed evaluation set for comparing the trained RL agent against heuristics.

Generates decision-sensitive Barabasi-Albert graph instances at the training regime's
(alpha, pfail, budget) point.  Instances where the degree vs random spread is below
EVAL_SPREAD_FILTER_DEGREE_RANDOM are discarded; the remaining ones are labelled with a
full 5-heuristic regime label and saved to eval_sets/rl_comparison.pkl.

Usage
-----
    python scripts/create_rl_comparison_eval_set.py [--config PATH] [--force]
                                                     [--num-graphs N] [--seeds-per-graph K]
                                                     [--output PATH]

Options
-------
--config PATH           Path to YAML config  (default: config/default.yaml)
--force                 Overwrite output file if it already exists
--num-graphs N          Candidate graphs to generate  (default: 50)
--seeds-per-graph K     Failure realisations per graph  (default: 5)
--output PATH           Output path  (default: eval_sets/rl_comparison.pkl)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
from cascading_rl.evaluation.saved_eval_sets import (
    DIAGNOSTIC_POLICY_NAMES,
    EVAL_SPREAD_FILTER_DEGREE_RANDOM,
    regime_label_from_heuristic_rollouts,
    rollout_final_anc_on_instance,
    save_eval_instances,
)
from cascading_rl.graph.generation import make_ba_graph

# Seed offset chosen to avoid overlap with ds_validation (uses training_seed + 50_000)
# and large-graph sets.  This set uses training_seed + 200_000.
_SEED_OFFSET = 200_000


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def _resolve_env_kwargs(config: dict) -> dict[str, object]:
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


def _resolve_budget_scaling(config: dict) -> tuple[int, bool]:
    training_graph = config["training"].get("graph", {})
    budget_scaling = config.get("budget_scaling", {})
    default_reference_n = int(training_graph.get("n_range", [30, 50])[1])
    reference_n_raw = budget_scaling.get("reference_n")
    reference_n = (
        int(reference_n_raw) if reference_n_raw is not None else default_reference_n
    )
    scale_budget = bool(budget_scaling.get("enabled", False))
    return reference_n, scale_budget


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed RL-vs-heuristic comparison eval set."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=50,
        help="Number of candidate graphs to generate (default: 50).",
    )
    parser.add_argument(
        "--seeds-per-graph",
        type=int,
        default=5,
        help="Number of failure seeds per graph (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval_sets" / "rl_comparison.pkl",
        help="Output path for the eval set (default: eval_sets/rl_comparison.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    out_path = args.output if args.output.is_absolute() else ROOT / args.output
    if out_path.exists() and not args.force:
        print(
            f"Eval set already exists at {out_path}.\n"
            "Pass --force to regenerate."
        )
        return

    config = _load_config(
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    training = config["training"]
    regime = training["regime"]
    regime_mapping = config["regime_mapping"]

    # --- Regime parameters from training config ---
    alpha: float = float(regime["alpha"])
    p_fail: float = float(regime["pfail"])
    b_ref: int = int(regime["budget"])
    n_ref, scale_budget = _resolve_budget_scaling(config)
    max_rounds: int = int(regime["max_rounds"])
    m: int = int(training["graph"]["m"])
    n_range: tuple[int, int] = tuple(training["graph"]["n_range"])  # type: ignore[assignment]

    env_kwargs = _resolve_env_kwargs(config)

    # --- Seeding ---
    # Use training seed + offset so this set is fully disjoint from ds_validation.
    master_seed: int = int(training["seed"]) + _SEED_OFFSET
    rng = Random(master_seed)

    print(f"Generating {args.num_graphs} graphs × {args.seeds_per_graph} seeds ...")
    print(
        f"Regime: alpha={alpha}, p_fail={p_fail}, b_ref={b_ref}, "
        f"n_range={n_range}, max_rounds={max_rounds}"
    )
    print(f"Spread filter: degree-random > {EVAL_SPREAD_FILTER_DEGREE_RANDOM:.2f}")

    # --- Build candidate (graph, meta) list ---
    graphs_meta: list[tuple[object, int, int]] = []
    for _ in range(args.num_graphs):
        n = rng.randint(n_range[0], n_range[1])
        graph_seed = rng.randint(0, 10**9)
        graphs_meta.append((make_ba_graph(n=n, m=m, seed=graph_seed), n, graph_seed))

    factories = build_policy_factories(base_seed=master_seed)
    base_failure_seed = master_seed + 1_000_000

    kept: list[dict] = []
    spreads: list[float] = []
    n_candidates = 0
    n_no_failure = 0
    n_low_spread = 0

    for gi, (graph, n, graph_seed) in enumerate(graphs_meta):
        b_scaled = compute_scaled_budget(
            b_ref,
            num_nodes=n,
            reference_n=n_ref,
            enabled=scale_budget,
        )
        for s in range(args.seeds_per_graph):
            n_candidates += 1
            failure_seed = base_failure_seed + gi * 1000 + s

            # Check that there are initial failures to recover from.
            probe = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=p_fail,
                budget=b_scaled,
                max_rounds=max_rounds,
                seed=0,
                **env_kwargs,
            )
            obs = probe.reset(seed=failure_seed)
            if not obs.failed:
                n_no_failure += 1
                continue

            initial_failures = frozenset(obs.failed)

            # Quick degree vs random spread filter.
            pol_degree = factories["degree"](gi, failure_seed)
            pol_random = factories["random"](gi, failure_seed)
            pr_degree = rollout_final_anc_on_instance(
                graph,
                alpha=alpha,
                p_fail=p_fail,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_degree,
            )
            pr_random = rollout_final_anc_on_instance(
                graph,
                alpha=alpha,
                p_fail=p_fail,
                budget=b_scaled,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_random,
            )
            spread = pr_degree - pr_random
            if spread <= EVAL_SPREAD_FILTER_DEGREE_RANDOM:
                n_low_spread += 1
                continue

            # Full 5-heuristic regime labelling.
            regime_label = regime_label_from_heuristic_rollouts(
                graph,
                alpha=alpha,
                p_fail=p_fail,
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

            spreads.append(spread)
            kept.append(
                {
                    "graph": graph,
                    "initial_failures": initial_failures,
                    "alpha": alpha,
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

        pct = (gi + 1) / len(graphs_meta) * 100
        print(f"  graph {gi + 1}/{len(graphs_meta)} ({pct:.0f}%)  kept so far: {len(kept)}", end="\r")

    print()  # newline after progress line

    if not kept:
        print(
            "WARNING: no instances passed the spread filter. "
            "Consider lowering --seeds-per-graph or adjusting the regime."
        )
        return

    save_eval_instances(out_path, kept)

    label_counts = Counter(inst["regime_label"] for inst in kept)
    mean_spread = sum(spreads) / len(spreads)

    print(f"\nWrote {len(kept)} instances to {out_path}")
    print(
        f"Candidates: {n_candidates}  "
        f"no-failure: {n_no_failure}  "
        f"low-spread: {n_low_spread}  "
        f"kept: {len(kept)}"
    )
    print(
        f"Spread: mean={mean_spread:.4f}  "
        f"min={min(spreads):.4f}  "
        f"max={max(spreads):.4f}"
    )
    print("Regime label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print(f"\nHeuristic policies used for labelling: {', '.join(DIAGNOSTIC_POLICY_NAMES)}")
    try:
        display_path = out_path.relative_to(ROOT)
    except ValueError:
        display_path = out_path
    print(
        "\nEvaluate with:\n"
        f"  python scripts/evaluate_policy.py --eval-set {display_path} "
        "--policies rl degree random risk greedy betweenness"
    )


if __name__ == "__main__":
    main()
