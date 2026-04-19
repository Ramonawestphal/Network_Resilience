"""Evaluate a trained checkpoint on Erdos-Renyi graphs.

This script runs the same evaluation protocol as evaluate_policy.py but
generates ER graphs instead of BA graphs, with edge probability p = 2m/n
to match the average degree of the BA training graphs.

Usage
-----
    python scripts/evaluate_er_graphs.py                        # default checkpoint
    python scripts/evaluate_er_graphs.py --checkpoint PATH      # specific checkpoint
    python scripts/evaluate_er_graphs.py --num-graphs 100 --seeds 0 1 2 3 4 5 6 7 8 9
    python scripts/evaluate_er_graphs.py --rl-only --output-dir experiments/eval_er_rl

Output is written to ``--output-dir`` as ``er_evaluation_summary.json`` and ``run_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
    summarize_episode_results,
)
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path
from scripts.reproducibility import write_run_metadata


def load_checkpoint(path: Path) -> tuple[RecoveryQNetwork, dict]:
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    from cascading_rl.models import QNetworkConfig
    config = QNetworkConfig(**data["model_config"])
    model = RecoveryQNetwork(config)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, data.get("training_config", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained policy on ER graphs.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "learner" / "recovery_q.pt",
        help="Path to the trained checkpoint (default: experiments/learner/recovery_q.pt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="YAML config file (used for regime parameters).",
    )
    parser.add_argument("--num-graphs", type=int, default=100, help="Number of ER graphs.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Failure seeds per graph.",
    )
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=999,
        help="Seed for graph generation (different from BA eval seed to avoid overlap).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "eval_er",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Evaluate only the learned RL policy (skip greedy, degree, betweenness, risk, random).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    training = config_data["training"]
    regime = training["regime"]
    budget_scaling = config_data.get("budget_scaling", {})

    alpha = float(regime["alpha"])
    pfail = float(regime["pfail"])
    budget = int(regime["budget"])
    max_rounds = int(regime["max_rounds"])
    m = int(training["graph"]["m"])
    n_range = tuple(training["graph"]["n_range"])
    scale_budget = bool(budget_scaling.get("enabled", True))
    scale_max_rounds = bool(budget_scaling.get("scale_max_rounds", True))
    reference_n = int(budget_scaling.get("reference_n", 40))

    print(f"Loading checkpoint from {args.checkpoint}")
    model, _ = load_checkpoint(args.checkpoint)

    import torch
    device = torch.device("cpu")
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)

    print(f"Generating {args.num_graphs} ER graphs (seed={args.graph_seed}, n_range={n_range}, m={m})")
    graphs = make_graph_batch(
        num_graphs=args.num_graphs,
        n_range=n_range,
        m=m,
        seed=args.graph_seed,
        graph_type="er",
    )

    baseline_factories = build_policy_factories(base_seed=0)
    policy_factories = {
        "rl": lambda gi, se: rl_policy,
        **baseline_factories,
    }

    print(f"Evaluating {len(policy_factories)} policies over {len(graphs)} graphs x {len(args.seeds)} seeds...")
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
        name: summarize_episode_results(episodes)
        for name, episodes in episodes_by_policy.items()
    }

    if args.rl_only:
        comparisons = []
    else:
        comparisons = compare_all_pairs(
            episodes_by_policy,
            baseline="degree",
            metric="anc_fixed",
            rng=__import__("random").Random(0),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _fmt(summary) -> dict:
        return {
            "anc_fixed_mean": summary.anc_fixed.mean,
            "anc_fixed_stderr": summary.anc_fixed.stderr,
            "final_nc_mean": summary.final_nc.mean,
            "final_nc_stderr": summary.final_nc.stderr,
            "solved_fraction_mean": summary.solved_fraction.mean,
            "rounds_mean": summary.rounds.mean,
            "episode_count": summary.episode_count,
        }

    result = {
        "graph_type": "er",
        "rl_only": args.rl_only,
        "policies": list(policy_factories.keys()),
        "checkpoint": portable_artifact_path(args.checkpoint),
        "num_graphs": args.num_graphs,
        "seeds": args.seeds,
        "regime": {"alpha": alpha, "pfail": pfail, "budget": budget, "max_rounds": max_rounds},
        "er_params": {"m": m, "n_range": list(n_range), "p_formula": "2*m/n"},
        "summaries": {name: _fmt(s) for name, s in summaries.items()},
        "comparisons_vs_degree": [
            {
                "policy": c.policy_a,
                "mean_diff_anc_fixed": c.mean_difference,
                "ci_low": c.bootstrap_ci_low,
                "ci_high": c.bootstrap_ci_high,
                "wilcoxon_p": c.wilcoxon_p_value,
                "significant": c.significant,
            }
            for c in comparisons
        ],
    }

    summary_path = args.output_dir / "er_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== ER Graph Evaluation Results ===")
    print(f"{'Policy':<14} {'ANC-fixed':>10} {'Solved':>8} {'Rounds':>8}")
    print("-" * 44)
    for name, s in summaries.items():
        print(
            f"{name:<14} {s.anc_fixed.mean:>10.3f} "
            f"{s.solved_fraction.mean:>8.3f} {s.rounds.mean:>8.1f}"
        )
    print(f"\nResults written to {summary_path}")

    write_run_metadata(
        args.output_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={
            "summary_path": portable_artifact_path(summary_path),
            "checkpoint_path": portable_artifact_path(args.checkpoint),
            "rl_only": args.rl_only,
        },
    )


if __name__ == "__main__":
    main()
