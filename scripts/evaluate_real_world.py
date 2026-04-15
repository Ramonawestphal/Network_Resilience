"""Evaluate a trained checkpoint on real-world network topologies.

Prerequisites
-------------
1. Download the datasets first:
       python scripts/download_real_world_data.py

2. Point ``--checkpoint`` at a trained policy (default: ``experiments/learner/recovery_q.pt``).
   Training is optional if you already have a checkpoint; this script only loads weights
   and runs rollouts (no training).

3. Run evaluation, e.g. RL-only with a distinct output directory:
       python scripts/evaluate_real_world.py --rl-only --output-dir experiments/eval_real_world_rl

Datasets
--------
ieee300 : IEEE 300-bus power transmission network (300 nodes, ~411 edges).
          Slightly larger than the BA training graphs (30-50 nodes) — tests
          whether the GNN generalises to larger, denser real infrastructure.

watts_strogatz : Watts-Strogatz small-world graph (n=300, k=4, p=0.1, seed=42).
          High clustering + short path lengths — structurally distinct from
          both BA (scale-free) and IEEE 300-bus (sparse tree-like).
          Reference: Watts & Strogatz (1998), Nature 393:440-442.

Method
------
Each real-world graph is treated as a single fixed topology. We vary only the
failure scenario (pfail, seed) across evaluation episodes. The cascade model
(load = degree, capacity = (1+alpha)*degree) is applied directly to the real
topology — this is a stylized approximation, not a physics-accurate simulation.
This is explicitly acknowledged in the output metadata.

Output
------
experiments/eval_real_world/<dataset>/evaluation_summary.json
experiments/eval_real_world/<dataset>/run_metadata.json
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
from cascading_rl.graph.generation import load_real_world_graph
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path
from scripts.reproducibility import write_run_metadata


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
        description="Evaluate trained policy on real-world network topologies."
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
        "--datasets",
        nargs="+",
        default=["ieee300"],
        help="Which datasets to evaluate (default: ieee300).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(20)),
        help="Failure seeds (default: 0..19, 20 seeds per graph).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Capacity slack override (default: from config training.regime.alpha).",
    )
    parser.add_argument(
        "--pfail",
        type=float,
        default=None,
        help="Failure rate override (default: from config training.regime.pfail).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Recovery budget override (default: from config training.regime.budget).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "eval_real_world",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Evaluate only the learned RL policy (skip greedy, degree, betweenness, risk, random).",
    )
    return parser.parse_args()


def _fmt_summary(summary) -> dict:
    return fmt_policy_summary(summary)


def evaluate_dataset(
    dataset_name: str,
    *,
    model: RecoveryQNetwork,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    seeds: list[int],
    output_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    reference_n: int = 40,
    scale_budget: bool = True,
    scale_max_rounds: bool = True,
    rl_only: bool = False,
) -> None:
    import torch
    print(f"\n{'='*55}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*55}")

    try:
        graph = load_real_world_graph(dataset_name)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    avg_degree = 2 * m / n

    from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds
    scaled_budget = compute_scaled_budget(budget, num_nodes=n, reference_n=reference_n, enabled=scale_budget)
    scaled_max_rounds = compute_scaled_max_rounds(max_rounds, num_nodes=n, reference_n=reference_n, enabled=scale_max_rounds)

    print(f"  Nodes: {n}  Edges: {m}  Avg degree: {avg_degree:.2f}")
    print(f"  Regime: alpha={alpha}, pfail={pfail}, budget={budget} -> scaled={scaled_budget}, "
          f"max_rounds={max_rounds} -> scaled={scaled_max_rounds}")
    print(f"  Seeds: {len(seeds)}")

    device = torch.device("cpu")
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)

    # Use sequential greedy for large graphs: exhaustive O(C(|failed|,k)) search
    # is infeasible once budget scaling pushes k beyond ~5.
    baseline_factories = build_policy_factories(base_seed=0, sequential_greedy=True)
    policy_factories = {
        "rl": lambda gi, se: rl_policy,
        **baseline_factories,
    }

    # Single graph — pass as a list of one
    print("  Running rollouts...", flush=True)
    episodes_by_policy = collect_matched_episodes(
        [graph],
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
        name: summarize_episode_results(episodes)
        for name, episodes in episodes_by_policy.items()
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

    out_dir = output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": dataset_name,
        "graph": {
            "num_nodes": n,
            "num_edges": m,
            "avg_degree": round(avg_degree, 3),
            "cascade_model_note": (
                "Load = node degree, capacity = (1+alpha)*degree. "
                "Stylized model — not physics-accurate for power grids."
            ),
        },
        "regime": {
            "alpha": alpha,
            "pfail": pfail,
            "budget_ref": budget,
            "budget_scaled": scaled_budget,
            "max_rounds_ref": max_rounds,
            "max_rounds_scaled": scaled_max_rounds,
            "reference_n": reference_n,
            "num_seeds": len(seeds),
        },
        "summaries": {name: _fmt_summary(s) for name, s in summaries.items()},
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

    summary_path = out_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Print full results table
    print(f"\n  {'Policy':<14} {'ANC-fix':>8} {'±se':>6} {'ANC-adp':>8} {'FinalNC':>8} "
          f"{'Solved':>7} {'Rounds':>7} {'ActRank':>8} {'NCgain':>8}")
    print(f"  {'-'*76}")
    policy_order = ["rl", "greedy", "degree", "betweenness", "risk", "random"]
    for name in policy_order:
        if name not in summaries:
            continue
        s = summaries[name]
        rws = f"{s.rounds_when_solved.mean:.1f}" if s.rounds_when_solved is not None else "  n/a"
        print(
            f"  {name:<14} {s.anc_fixed.mean:>8.3f} {s.anc_fixed.stderr:>6.3f} "
            f"{s.anc_adaptive.mean:>8.3f} {s.final_nc.mean:>8.3f} "
            f"{s.solved_fraction.mean:>7.3f} {rws:>7} "
            f"{s.mean_action_rank.mean:>8.2f} {s.mean_nc_gain.mean:>8.4f}"
        )

    print(f"\n  Saved -> {summary_path}")
    write_run_metadata(
        out_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=config_path,
        extra={
            "dataset": dataset_name,
            "summary_path": portable_artifact_path(summary_path),
            "checkpoint_path": portable_artifact_path(checkpoint_path),
            "rl_only": rl_only,
        },
    )


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    regime = cfg["training"]["regime"]
    budget_scaling = cfg.get("budget_scaling", {})
    alpha = args.alpha if args.alpha is not None else float(regime["alpha"])
    pfail = args.pfail if args.pfail is not None else float(regime["pfail"])
    budget = args.budget if args.budget is not None else int(regime["budget"])
    max_rounds = int(regime["max_rounds"])
    reference_n = int(budget_scaling.get("reference_n", 40))
    scale_budget = bool(budget_scaling.get("enabled", True))
    scale_max_rounds = bool(budget_scaling.get("scale_max_rounds", True))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)

    for dataset in args.datasets:
        evaluate_dataset(
            dataset,
            model=model,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            seeds=args.seeds,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            reference_n=reference_n,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            rl_only=args.rl_only,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
