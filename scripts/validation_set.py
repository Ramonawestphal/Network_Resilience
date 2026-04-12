#!/usr/bin/env python3
"""Generate ``eval_sets/validation_set.json``: 50 BA graphs (|V| in 30–50) at alpha=0.25, p_fail=0.2, B=2.

Only instances where **greedy** and **random** baseline policies differ meaningfully in final NC
(``|pr_greedy - pr_random| >= MIN_NC_DIFF``) are kept.

Run from repo root::

    python scripts/validation_set.py

Output: ``eval_sets/validation_set.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation.regime import build_policy_factories
from cascading_rl.evaluation.saved_eval_sets import rollout_final_nc_on_instance, save_eval_instances
from cascading_rl.graph.generation import make_ba_graph

OUTPUT_JSON = ROOT / "eval_sets" / "validation_set.json"

NUM_INSTANCES = 50
ALPHA = 0.25
P_FAIL = 0.2
BUDGET = 2
N_RANGE = (30, 50)
# Minimum |final_nc_greedy − final_nc_random| to count as “perform differently”.
MIN_NC_DIFF = 0.005
MASTER_SEED = 7 + 91_000
MAX_SEED_TRIES_PER_GRAPH = 400
MAX_GRAPH_ATTEMPTS = 8_000


def _default_env_and_graph_settings() -> tuple[dict[str, object], int, int, int]:
    """Match ``config/default.yaml`` training / graph / budget_scaling (keep in sync)."""
    env_kwargs: dict[str, object] = {
        "capacity_noise": 0.0,
        "failure_bias": "uniform",
        "action_space": "failed",
        "obs_hops": None,
        "abandonment_nc_threshold": None,
    }
    m = 2
    max_rounds = 20
    n_ref = 40
    return env_kwargs, m, max_rounds, n_ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite validation_set.json if it exists.",
    )
    args = parser.parse_args()

    if OUTPUT_JSON.exists() and not args.force:
        print(
            f"{OUTPUT_JSON} already exists. Pass --force to regenerate.",
            file=sys.stderr,
        )
        sys.exit(1)

    env_kwargs, m, max_rounds, n_ref = _default_env_and_graph_settings()
    rng = Random(MASTER_SEED)
    factories = build_policy_factories(base_seed=MASTER_SEED)

    kept: list[dict[str, object]] = []
    graphs_tried = 0

    while len(kept) < NUM_INSTANCES:
        if graphs_tried >= MAX_GRAPH_ATTEMPTS:
            raise RuntimeError(
                f"Stopped after {MAX_GRAPH_ATTEMPTS} graphs with only {len(kept)} valid instances. "
                f"Try lowering MIN_NC_DIFF (currently {MIN_NC_DIFF}) or increasing MAX_GRAPH_ATTEMPTS."
            )
        graphs_tried += 1
        n = rng.randint(N_RANGE[0], N_RANGE[1])
        graph_seed = rng.randint(0, 10**9)
        graph = make_ba_graph(n=n, m=m, seed=graph_seed)
        graph_index = len(kept)

        for _ in range(MAX_SEED_TRIES_PER_GRAPH):
            failure_seed = rng.randint(0, 10**9)
            probe = RecoveryEnv(
                graph,
                alpha=ALPHA,
                pfail=P_FAIL,
                budget=BUDGET,
                max_rounds=max_rounds,
                seed=0,
                **env_kwargs,
            )
            obs = probe.reset(seed=failure_seed)
            if not obs.failed:
                continue

            initial_failures = frozenset(obs.failed)
            pol_greedy = factories["greedy"](graph_index, failure_seed)
            pol_random = factories["random"](graph_index, failure_seed)
            pr_greedy = rollout_final_nc_on_instance(
                graph,
                alpha=ALPHA,
                p_fail=P_FAIL,
                budget=BUDGET,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_greedy,
            )
            pr_random = rollout_final_nc_on_instance(
                graph,
                alpha=ALPHA,
                p_fail=P_FAIL,
                budget=BUDGET,
                max_rounds=max_rounds,
                failure_seed=failure_seed,
                env_kwargs=env_kwargs,
                policy=pol_random,
            )
            if abs(pr_greedy - pr_random) < MIN_NC_DIFF:
                continue

            kept.append(
                {
                    "graph": graph,
                    "initial_failures": initial_failures,
                    "alpha": ALPHA,
                    "p_fail": P_FAIL,
                    "budget": BUDGET,
                    "b_scaled": BUDGET,
                    "b_ref": BUDGET,
                    "n_ref": n_ref,
                    "n": n,
                    "graph_seed": graph_seed,
                    "failure_seed": failure_seed,
                    "pr_greedy": pr_greedy,
                    "pr_random": pr_random,
                    "spread": pr_greedy - pr_random,
                    "max_rounds": max_rounds,
                    "m": m,
                    "regime_label": "greedy_vs_random",
                }
            )
            break

    save_eval_instances(OUTPUT_JSON, kept)
    print(f"Wrote {len(kept)} instances to {OUTPUT_JSON}")
    spreads = [float(x["spread"]) for x in kept]  # type: ignore[misc]
    print(
        f"spread stats: min={min(spreads):.4f} max={max(spreads):.4f} "
        f"mean={sum(spreads) / len(spreads):.4f}"
    )


if __name__ == "__main__":
    main()
