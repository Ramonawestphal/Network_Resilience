"""Evaluate all policies across cascade parameters on in-distribution graphs.

Sweeps a (alpha, pfail, budget) grid on 100 BA graphs with n drawn uniformly
from [20, 50]. The same graph set is reused across all parameter cells so
differences isolate the cascade dynamics rather than graph structure.

All 6 policies are evaluated: rl, greedy, degree, betweenness, risk, random.

Grid (defaults, all overridable via CLI)
----------------------------------------
  alpha   : [0.10, 0.20, 0.25, 0.30]  -- low / mid / train / high capacity slack
  pfail   : [0.05, 0.15, 0.20, 0.25]  -- low / mid / train / high failure rate
  budget  : [1, 2, 3, 4]              -- budget=2 is training budget

Total default cells : 4 x 4 x 3 = 48
Training cell       : alpha=0.25, pfail=0.20, budget=2  (flagged is_training_params=True)
Graphs              : 100, n ~ Uniform[20, 50] (shared across all cells)
Seeds per graph     : 10
Episodes per cell   : 1 000

Output
------
experiments/eval_param_generalization/param_generalization_summary.json
experiments/eval_param_generalization/run_metadata.json

Usage
-----
    python scripts/evaluate_param_generalization.py
    python scripts/evaluate_param_generalization.py \
        --alpha 0.20 0.25 --pfail 0.10 0.20 --budget 1 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from random import Random

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation.benchmarks import (
    build_policy_factories,
    collect_matched_episodes,
    compare_all_pairs,
    fmt_policy_summary,
    summarize_episode_results,
)
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path
from scripts.reproducibility import write_run_metadata

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

N_LOW = 30
N_HIGH = 50
DEFAULT_ALPHA = [0.10, 0.20, 0.25, 0.30]   # 0.25 = training alpha
DEFAULT_PFAIL = [0.05, 0.15, 0.20, 0.25]   # 0.20 = training pfail
DEFAULT_BUDGET = [1, 2, 3]                  # budget=2 = training budget
DEFAULT_NUM_GRAPHS = 40
DEFAULT_SEEDS = list(range(5))

POLICY_PRINT_ORDER = ["rl", "greedy", "degree", "betweenness", "risk", "random"]


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> RecoveryQNetwork:
    import torch
    from cascading_rl.models import QNetworkConfig
    data = torch.load(path, map_location="cpu", weights_only=False)
    config = QNetworkConfig(**data["model_config"])
    model = RecoveryQNetwork(config)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_GRAPH_SEEDS = {"ba": 7777, "er": 8888, "ws": 9999}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all policies across (alpha, pfail, budget) on a chosen graph type."
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "experiments" / "learner" / "recovery_q.pt")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config" / "default.yaml")
    parser.add_argument("--graph-type", type=str, default="ba",
                        choices=["ba", "er", "ws"],
                        help="Graph topology to generate (default: ba).")
    parser.add_argument("--n-low", type=int, default=N_LOW,
                        help="Lower bound for uniform graph-size draw (default: 30).")
    parser.add_argument("--n-high", type=int, default=N_HIGH,
                        help="Upper bound for uniform graph-size draw (default: 50).")
    parser.add_argument("--alpha", type=float, nargs="+", default=DEFAULT_ALPHA)
    parser.add_argument("--pfail", type=float, nargs="+", default=DEFAULT_PFAIL)
    parser.add_argument("--budget", type=int, nargs="+", default=DEFAULT_BUDGET)
    parser.add_argument("--num-graphs", type=int, default=DEFAULT_NUM_GRAPHS,
                        help="Graphs shared across all cells (default: 100).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Failure seeds per graph (default: 0..9).")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "eval_param_generalization")
    parser.add_argument("--sequential-greedy", action="store_true",
                        help="Use sequential O(|failed|*k) greedy instead of exhaustive "
                             "O(C(|failed|,k)) search. Required for large graphs.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_comparisons(comparisons: list) -> list[dict]:
    return [
        {
            "policy": c.policy_a,
            "mean_diff_anc_fixed": round(c.mean_difference, 4),
            "ci_95_low": round(c.bootstrap_ci_low, 4),
            "ci_95_high": round(c.bootstrap_ci_high, 4),
            "wilcoxon_p": round(c.wilcoxon_p_value, 4),
            "significant_p005": c.significant,
        }
        for c in comparisons
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training = cfg["training"]
    regime = training["regime"]
    budget_scaling = cfg.get("budget_scaling", {})

    train_alpha = float(regime["alpha"])
    train_pfail = float(regime["pfail"])
    train_budget = int(regime["budget"])
    max_rounds = int(regime["max_rounds"])
    m = int(training["graph"]["m"])
    scale_budget = bool(budget_scaling.get("enabled", True))
    scale_max_rounds = bool(budget_scaling.get("scale_max_rounds", True))
    reference_n = int(budget_scaling.get("reference_n", 40))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)

    import torch
    device = torch.device("cpu")
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)

    total_cells = len(args.alpha) * len(args.pfail) * len(args.budget)
    episodes_per_cell = args.num_graphs * len(args.seeds)

    print(f"\nGrid: {len(args.alpha)} alpha x {len(args.pfail)} pfail x "
          f"{len(args.budget)} budget = {total_cells} cells")
    print(f"Graph type: {args.graph_type.upper()}  n ~ Uniform[{args.n_low}, {args.n_high}]  "
          f"{args.num_graphs} graphs  {len(args.seeds)} seeds -> {episodes_per_cell} episodes/cell")
    print(f"Training reference: alpha={train_alpha}, pfail={train_pfail}, budget={train_budget}")

    # Generate one shared graph set reused across all (alpha, pfail, budget) cells.
    # Each graph type uses a distinct seed so pools don't overlap when all three
    # are run back-to-back.
    graph_seed = _GRAPH_SEEDS.get(args.graph_type, hash(args.graph_type) % 10**6)
    graphs = make_graph_batch(
        num_graphs=args.num_graphs,
        n_range=(args.n_low, args.n_high),
        m=m,
        seed=graph_seed,
        graph_type=args.graph_type,
    )
    for i, g in enumerate(graphs):
        g.graph["graph_index"] = i

    sizes = [g.number_of_nodes() for g in graphs]
    avg_n = sum(sizes) / len(sizes)
    avg_deg = sum(2 * g.number_of_edges() / g.number_of_nodes() for g in graphs) / len(graphs)
    print(f"Generated {args.num_graphs} graphs: n in [{min(sizes)}, {max(sizes)}] "
          f"mean_n={avg_n:.1f}  avg_degree={avg_deg:.2f}")

    if args.sequential_greedy:
        print("Greedy: sequential approximation O(|failed|*k)  [exhaustive search disabled]")
    policy_factories = {
        "rl": lambda gi, se: rl_policy,
        **build_policy_factories(base_seed=0, sequential_greedy=args.sequential_greedy),
    }

    cells: list[dict] = []

    for cell_idx, (alpha, pfail, budget) in enumerate(
        itertools.product(args.alpha, args.pfail, args.budget), start=1
    ):
        is_train = (alpha == train_alpha and pfail == train_pfail and budget == train_budget)
        tag = " [TRAIN]" if is_train else ""
        print(f"\n[{cell_idx}/{total_cells}] alpha={alpha}  pfail={pfail}  budget={budget}{tag}",
              flush=True)

        episodes_by_policy = collect_matched_episodes(
            graphs,
            policy_factories,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            seeds=args.seeds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
        )

        summaries = {
            name: summarize_episode_results(eps)
            for name, eps in episodes_by_policy.items()
        }
        comparisons = compare_all_pairs(
            episodes_by_policy,
            baseline="degree",
            metric="anc_fixed",
            rng=Random(0),
        )

        print(f"  {'Policy':<14} {'ANC-fix':>8} {'+-se':>6} {'ANC-adp':>8} {'FinalNC':>8} "
              f"{'Solved':>7} {'Rounds':>7} {'ActRank':>8} {'NCgain':>8}")
        print(f"  {'-'*76}")
        for name in POLICY_PRINT_ORDER:
            if name not in summaries:
                continue
            s = summaries[name]
            rws = f"{s.rounds_when_solved.mean:.1f}" if s.rounds_when_solved is not None else "  n/a"
            print(f"  {name:<14} {s.anc_fixed.mean:>8.3f} {s.anc_fixed.stderr:>6.3f} "
                  f"{s.anc_adaptive.mean:>8.3f} {s.final_nc.mean:>8.3f} "
                  f"{s.solved_fraction.mean:>7.3f} {rws:>7} "
                  f"{s.mean_action_rank.mean:>8.2f} {s.mean_nc_gain.mean:>8.4f}")

        cells.append({
            "alpha": alpha,
            "pfail": pfail,
            "budget": budget,
            "is_training_params": is_train,
            "episode_count": episodes_per_cell,
            "summaries": {name: fmt_policy_summary(s) for name, s in summaries.items()},
            "comparisons_vs_degree": _fmt_comparisons(comparisons),
        })

    output = {
        "description": (
            f"Parameter generalisation sweep: all 6 policies across (alpha, pfail, budget) grid "
            f"on {args.num_graphs} {args.graph_type.upper()} graphs with "
            f"n ~ Uniform[{args.n_low}, {args.n_high}]. "
            "Same graph set reused across all cells."
        ),
        "training_reference": {
            "alpha": train_alpha,
            "pfail": train_pfail,
            "budget": train_budget,
        },
        "grid_spec": {
            "graph_type": args.graph_type,
            "alpha_values": args.alpha,
            "pfail_values": args.pfail,
            "budget_values": args.budget,
            "total_cells": total_cells,
            "num_graphs": args.num_graphs,
            "n_low": args.n_low,
            "n_high": args.n_high,
            "num_seeds": len(args.seeds),
            "episodes_per_cell": episodes_per_cell,
            "scale_budget": scale_budget,
            "scale_max_rounds": scale_max_rounds,
            "reference_n": reference_n,
            "m": m,
        },
        "graph_set": {
            "n_min": min(sizes),
            "n_max": max(sizes),
            "n_mean": round(avg_n, 1),
            "avg_degree": round(avg_deg, 2),
        },
        "cells": cells,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "param_generalization_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved -> {summary_path}")
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={"summary_path": portable_artifact_path(summary_path)},
    )
    print(f"\nDone. {total_cells} cells x {episodes_per_cell} episodes = "
          f"{total_cells * episodes_per_cell} total episodes.")


if __name__ == "__main__":
    main()
