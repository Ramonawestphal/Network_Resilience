"""Evaluate trained policy on larger BA graphs: n=100, 200, 500, 1000.

Tests scale generalisation beyond the training range (n in [30, 50]).
Budget is scaled proportionally to graph size (same reference_n=40 as training).
Each graph size uses 20 graphs x 10 failure seeds = 200 episodes (defaults).

Without ``--rl-only``, greedy and other heuristics are included. Exhaustive greedy
can be very slow at large n and high budgets; use ``--exclude-greedy`` to keep RL and
all other baselines but skip greedy, or ``--rl-only`` for fast runs that only score
the learned policy.

Output
------
experiments/eval_larger_ba/larger_ba_summary.json
experiments/eval_larger_ba/run_metadata.json

Usage
-----
    python scripts/evaluate_larger_ba.py
    python scripts/evaluate_larger_ba.py --sizes 100 200 500 1000 --num-graphs 20
    python scripts/evaluate_larger_ba.py --exclude-greedy
    python scripts/evaluate_larger_ba.py --rl-only --output-dir experiments/eval_larger_ba_rl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path
from scripts.reproducibility import write_run_metadata

POLICY_ORDER = ["rl", "greedy", "degree", "betweenness", "risk", "random"]


def load_checkpoint(path: Path) -> RecoveryQNetwork:
    import torch
    from cascading_rl.models import QNetworkConfig
    data = torch.load(path, map_location="cpu", weights_only=False)
    config = QNetworkConfig(**data["model_config"])
    model = RecoveryQNetwork(config)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained policy on larger BA graphs (scale generalisation)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "learner" / "recovery_q.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="Exact graph sizes to evaluate (default: 100 200 500, 1000).",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=20,
        help="Graphs per size (default: 20).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Failure seeds per graph (default: 0..9).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "eval_larger_ba",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Evaluate only the learned RL policy (skip greedy, degree, betweenness, risk, random).",
    )
    parser.add_argument(
        "--exclude-greedy",
        action="store_true",
        dest="exclude_greedy",
        help="With RL + heuristics, skip the exhaustive NC-greedy baseline (slow at large n). "
             "Incompatible with --rl-only.",
    )
    args = parser.parse_args()
    if args.rl_only and args.exclude_greedy:
        parser.error("Use only one of --rl-only and --exclude-greedy.")
    return args


def _fmt(summary) -> dict:
    return fmt_policy_summary(summary)


def run_size(
    n: int,
    *,
    model: RecoveryQNetwork,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    m: int,
    num_graphs: int,
    seeds: list[int],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
    rl_only: bool = False,
    exclude_greedy: bool = False,
) -> tuple[dict, list]:
    import torch
    from random import Random

    print(f"\n{'='*55}")
    print(f"Graph size: n = {n}")
    print(f"{'='*55}")

    rng = Random(5000 + n)   # deterministic, separate from training/eval seeds
    graphs = []
    for i in range(num_graphs):
        g = make_ba_graph(n=n, m=m, seed=rng.randint(0, 10**9))
        g.graph["graph_index"] = i
        graphs.append(g)

    avg_deg = sum(2 * g.number_of_edges() / g.number_of_nodes() for g in graphs) / len(graphs)
    print(f"  Graphs: {num_graphs}  n={n}  avg_degree={avg_deg:.2f}")

    device = torch.device("cpu")
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)
    if rl_only:
        policy_factories = {"rl": lambda gi, se: rl_policy}
    else:
        baseline_factories = build_policy_factories(base_seed=0)
        if exclude_greedy:
            baseline_factories = {k: v for k, v in baseline_factories.items() if k != "greedy"}
        policy_factories = {
            "rl": lambda gi, se: rl_policy,
            **baseline_factories,
        }

    print(f"  Running {len(policy_factories)} policies x {num_graphs} graphs x {len(seeds)} seeds...", flush=True)
    episodes_by_policy = collect_matched_episodes(
        graphs,
        policy_factories,
        alpha=alpha,
        pfail=pfail,
        budget=budget,
        max_rounds=max_rounds,
        seeds=seeds,
        scale_budget=scale_budget,
        scale_max_rounds=scale_max_rounds,
        reference_n=reference_n,
    )

    summaries = {
        name: summarize_episode_results(eps)
        for name, eps in episodes_by_policy.items()
    }

    if rl_only:
        comparisons = []
    else:
        comparisons = compare_all_pairs(
            episodes_by_policy,
            baseline="degree",
            metric="anc_fixed",
            rng=__import__("random").Random(0),
        )

    print(f"\n  {'Policy':<14} {'ANC-fixed':>10} {'+/-se':>8} {'Solved':>8} {'Rounds':>7}")
    print(f"  {'-'*50}")
    for name in POLICY_ORDER:
        if name not in summaries:
            continue
        s = summaries[name]
        print(
            f"  {name:<14} {s.anc_fixed.mean:>10.3f} "
            f"{s.anc_fixed.stderr:>8.3f} "
            f"{s.solved_fraction.mean:>8.3f} "
            f"{s.rounds.mean:>7.1f}"
        )

    return summaries, comparisons


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training = cfg["training"]
    regime = training["regime"]
    budget_scaling = cfg.get("budget_scaling", {})

    alpha = float(regime["alpha"])
    pfail = float(regime["pfail"])
    budget = int(regime["budget"])
    max_rounds = int(regime["max_rounds"])
    m = int(training["graph"]["m"])
    scale_budget = bool(budget_scaling.get("enabled", True))
    scale_max_rounds = bool(budget_scaling.get("scale_max_rounds", True))
    reference_n = int(budget_scaling.get("reference_n", 40))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)

    print(f"\nRegime: alpha={alpha}, pfail={pfail}, budget={budget} (scaled), max_rounds={max_rounds}")
    print(f"Sizes: {args.sizes}  |  num_graphs={args.num_graphs}  |  seeds={args.seeds}")
    print(f"Training range: n in [30, 50]  ->  all sizes are OOD")

    results_by_size: dict[str, dict] = {}

    for n in args.sizes:
        summaries, comparisons = run_size(
            n,
            model=model,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            m=m,
            num_graphs=args.num_graphs,
            seeds=args.seeds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
            rl_only=args.rl_only,
            exclude_greedy=args.exclude_greedy,
        )
        results_by_size[str(n)] = {
            "n": n,
            "summaries": {name: _fmt(s) for name, s in summaries.items()},
            "comparisons_vs_degree": [
                {
                    "policy": c.policy_a,
                    "mean_diff_anc_fixed": round(c.mean_difference, 4),
                    "ci_95_low": round(c.bootstrap_ci_low, 4),
                    "ci_95_high": round(c.bootstrap_ci_high, 4),
                    "wilcoxon_p": round(c.wilcoxon_p_value, 4),
                    "significant_p005": c.significant,
                }
                for c in comparisons
            ],
        }

    if args.rl_only:
        policy_names = ["rl"]
    elif args.exclude_greedy:
        policy_names = ["rl", "degree", "betweenness", "risk", "random"]
    else:
        policy_names = ["rl", "greedy", "degree", "betweenness", "risk", "random"]

    output = {
        "description": "Scale generalisation: BA graphs at n=100, 200, 500 (all OOD)",
        "rl_only": args.rl_only,
        "exclude_greedy": args.exclude_greedy,
        "policies": policy_names,
        "checkpoint": portable_artifact_path(args.checkpoint),
        "training_range": [30, 50],
        "regime": {
            "alpha": alpha,
            "pfail": pfail,
            "budget_ref": budget,
            "budget_scaled": True,
            "reference_n": reference_n,
            "max_rounds": max_rounds,
        },
        "graph_params": {"m": m, "num_graphs_per_size": args.num_graphs, "num_seeds": len(args.seeds)},
        "sizes": results_by_size,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "larger_ba_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved -> {summary_path}")
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={
            "summary_path": portable_artifact_path(summary_path),
            "checkpoint_path": portable_artifact_path(args.checkpoint),
            "rl_only": args.rl_only,
            "exclude_greedy": args.exclude_greedy,
        },
    )
    print("\nAll done.")


if __name__ == "__main__":
    main()
