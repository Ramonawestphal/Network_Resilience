from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, cast

import torch
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random import Random

from cascading_rl.envs.recovery import RecoveryObservation
from cascading_rl.evaluation import (
    build_policy_factories,
    estimate_minimum_budget,
    evaluate_policy_factories_on_graphs,
)
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network
from cascading_rl.policies import choose_random_failed_node

CHECKPOINT_PATH = ROOT / "experiments" / "learner" / "recovery_q.pt"
ALPHA_GRID = (0.05, 0.1)
PFAIL_GRID = (0.10, 0.15, 0.20)
SEEDS = tuple(range(10))
NUM_GRAPHS = 30


PolicyFn = Callable[[RecoveryObservation], object]
PolicyFactory = Callable[[int, int], PolicyFn]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristics and RL in hard cascade regimes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="YAML config (evaluation.tau, evaluation.budgets, hard_regime).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hard = config["hard_regime"]
    evaluation = config["evaluation"]
    tau = float(evaluation["tau"])
    evaluation_budgets = list(evaluation["budgets"])
    n_range = tuple(hard["n_range"])
    m = int(hard["m"])
    budget = int(hard["budget"])
    max_rounds = int(hard["max_rounds"])

    output_dir = ROOT / "experiments" / "hard_regime"
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs = make_graph_batch(
        num_graphs=NUM_GRAPHS,
        n_range=n_range,
        m=m,
        seed=4242,
    )
    representative_graph = graphs[0]
    base_factories = build_policy_factories(base_seed=0)
    heuristic_names = ("random", "degree", "risk", "greedy", "betweenness")
    device = torch.device("cpu")
    num_seeds = len(SEEDS)

    summary_rows: list[tuple[float, float, str, dict[str, float], float | None]] = []

    for alpha in ALPHA_GRID:
        for pfail in PFAIL_GRID:
            policy_factories: dict[str, PolicyFactory] = {
                name: base_factories[name] for name in heuristic_names
            }
            rl_policy = None
            if CHECKPOINT_PATH.exists():
                model, _ = load_q_network(CHECKPOINT_PATH, map_location=device)
                rl_policy = build_greedy_policy(model, device=device)
                policy_factories["rl"] = lambda _gi, _se: rl_policy

            summaries = evaluate_policy_factories_on_graphs(
                graphs,
                policy_factories,
                alpha=float(alpha),
                pfail=float(pfail),
                budget=budget,
                max_rounds=max_rounds,
                seeds=SEEDS,
                tau=tau,
            )

            serialized: dict[str, dict[str, object]] = {
                policy_name: {
                    "final_anc_mean": summary.final_anc.mean,
                    "final_anc_stderr": summary.final_anc.stderr,
                    "threshold_hit_mean": summary.threshold_hit_fraction.mean,
                    "rounds_mean": summary.rounds.mean,
                    "solved_fraction_mean": summary.solved_fraction.mean,
                }
                for policy_name, summary in summaries.items()
            }

            for policy_name in serialized:
                if policy_name == "random":
                    pol_eval = lambda observation: choose_random_failed_node(observation, rng=Random(0))
                elif policy_name == "rl":
                    assert rl_policy is not None
                    pol_eval = rl_policy
                else:
                    pol_eval = base_factories[policy_name](0, 0)
                serialized[policy_name]["b_star"] = estimate_minimum_budget(
                    representative_graph,
                    pol_eval,
                    tau=tau,
                    budgets=evaluation_budgets,
                    trials=num_seeds,
                    alpha=float(alpha),
                    pfail=float(pfail),
                    max_rounds=max_rounds,
                )[0]

            out_path = output_dir / f"results_{alpha:.2f}_{pfail:.2f}.json"
            with out_path.open("w", encoding="utf-8") as file:
                json.dump(serialized, file, indent=2)

            means: dict[str, float] = {
                n: cast(float, serialized[n]["final_anc_mean"]) for n in serialized
            }
            winner = max(means, key=lambda k: means[k])
            gap_rl_greedy: float | None = None
            if "rl" in means and "greedy" in means:
                gap_rl_greedy = means["rl"] - means["greedy"]
            summary_rows.append((alpha, pfail, winner, means, gap_rl_greedy))

    all_policies = sorted({k for _a, _p, _w, mns, _g in summary_rows for k in mns})
    print("Hard-regime evaluation — final_anc_mean per policy")
    header = "alpha\tpfail\twinner\t" + "\t".join(all_policies) + "\tRL_minus_greedy"
    print(header)
    for alpha, pfail, winner, means, gap in summary_rows:
        vals = "\t".join(f"{means.get(p, float('nan')):.3f}" for p in all_policies)
        gap_s = f"{gap:.3f}" if gap is not None else "n/a"
        print(f"{alpha}\t{pfail}\t{winner}\t{vals}\t{gap_s}")
    print(f"Saved results under {output_dir}")


if __name__ == "__main__":
    main()
