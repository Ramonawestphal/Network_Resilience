"""Topology ablation: BA vs ER vs WS at matched scale and degree.

Tier 1 of the two-tier evaluation structure:

    Tier 1 — Topology ablation (n ∈ [30, 50], matched average degree)
        BA  : Barabási-Albert, scale-free, m=2 → avg degree ≈ 4  (training distribution)
        ER  : Erdős-Rényi,     random,     p = 2m/n → avg degree ≈ 4
        WS  : Watts-Strogatz,  small-world, k=4, p=0.1 → avg degree = 4

    Tier 2 — OOD evaluation (n = 300, real/realistic topologies)
        See scripts/evaluate_real_world.py

All three graph types use the same n_range, m, and failure regime drawn from
config/default.yaml. Graphs are generated with distinct seeds so BA/ER/WS sets
do not overlap; the episode seed lists are identical across types so comparisons
are matched on failure scenario.

Output
------
experiments/eval_topology_ablation/topology_ablation_summary.json
experiments/eval_topology_ablation/run_metadata.json

Usage
-----
    python scripts/evaluate_topology_ablation.py
    python scripts/evaluate_topology_ablation.py --num-graphs 50 --seeds 0 1 2 3 4
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
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path
from scripts.reproducibility import write_run_metadata

# Distinct graph-generation seeds for each topology type so the graph pools
# do not overlap.  Episode failure seeds are shared across types.
_GRAPH_SEEDS = {"ba": 0, "er": 999, "ws": 1999}

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
        description="Topology ablation: BA vs ER vs WS at n∈[30,50], matched degree."
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
        "--topologies",
        nargs="+",
        default=["ba", "er", "ws"],
        help="Which topology types to include (default: ba er ws).",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=100,
        help="Number of graphs per topology type (default: 100).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Failure seeds per graph (default: 0..9).",
    )
    parser.add_argument("--alpha", type=float, default=None,
                        help="Capacity slack override (default: from config).")
    parser.add_argument("--pfail", type=float, default=None,
                        help="Failure rate override (default: from config).")
    parser.add_argument("--budget", type=int, default=None,
                        help="Recovery budget override (default: from config).")
    parser.add_argument("--n-range", type=int, nargs=2, default=[30, 50],
                        metavar=("N_LOW", "N_HIGH"),
                        help="Graph size range for all topologies (default: 30 50).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "eval_topology_ablation",
    )
    return parser.parse_args()


def _fmt_summary(summary) -> dict:
    return fmt_policy_summary(summary)


def run_topology(
    topology: str,
    *,
    model: RecoveryQNetwork,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    m: int,
    n_range: tuple[int, int],
    num_graphs: int,
    seeds: list[int],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
) -> tuple[dict, list]:
    """Run all policies on one topology type. Returns (summaries_dict, comparisons_list)."""
    import torch
    print(f"\n{'='*55}")
    print(f"Topology: {topology.upper()}")
    print(f"{'='*55}")

    graph_seed = _GRAPH_SEEDS.get(topology, hash(topology) % 10**6)
    graphs = make_graph_batch(
        num_graphs=num_graphs,
        n_range=n_range,
        m=m,
        seed=graph_seed,
        graph_type=topology,
    )
    avg_n = sum(g.number_of_nodes() for g in graphs) / len(graphs)
    avg_deg = sum(2 * g.number_of_edges() / g.number_of_nodes() for g in graphs) / len(graphs)
    print(f"  Graphs: {len(graphs)}  avg_n={avg_n:.1f}  avg_degree={avg_deg:.2f}")

    device = torch.device("cpu")
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)
    baseline_factories = build_policy_factories(base_seed=0)
    policy_factories = {
        "rl": lambda gi, se: rl_policy,
        **baseline_factories,
    }

    print(f"  Running {len(policy_factories)} policies × {len(graphs)} graphs × {len(seeds)} seeds...", flush=True)
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

    comparisons = compare_all_pairs(
        episodes_by_policy,
        baseline="degree",
        metric="anc_fixed",
        rng=__import__("random").Random(0),
    )

    # Print regime + full results table
    print(f"  Regime: alpha={alpha}  pfail={pfail}  budget={budget}  max_rounds={max_rounds}")
    print(f"  n_range={n_range}  m={m}  scale_budget={scale_budget}  reference_n={reference_n}")
    print(f"\n  {'Policy':<14} {'ANC-fix':>8} {'±se':>6} {'ANC-adp':>8} {'FinalNC':>8} "
          f"{'Solved':>7} {'Rounds':>7} {'ActRank':>8} {'NCgain':>8}")
    print(f"  {'-'*76}")
    for name in POLICY_ORDER:
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

    return summaries, comparisons


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training = cfg["training"]
    regime = training["regime"]
    budget_scaling = cfg.get("budget_scaling", {})

    alpha = args.alpha if args.alpha is not None else float(regime["alpha"])
    pfail = args.pfail if args.pfail is not None else float(regime["pfail"])
    budget = args.budget if args.budget is not None else int(regime["budget"])
    max_rounds = int(regime["max_rounds"])
    m = int(training["graph"]["m"])
    n_range = tuple(args.n_range)  # default [30, 50]; overridable via --n-range
    scale_budget = bool(budget_scaling.get("enabled", True))
    scale_max_rounds = bool(budget_scaling.get("scale_max_rounds", True))
    reference_n = int(budget_scaling.get("reference_n", 40))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)

    print(f"\nRegime: alpha={alpha}, pfail={pfail}, budget={budget}, max_rounds={max_rounds}")
    print(f"Graph params: n_range={n_range}, m={m} (avg degree ≈ {2*m})")
    print(f"Topologies: {args.topologies}  |  num_graphs={args.num_graphs}  |  seeds={args.seeds}")

    results_by_topology: dict[str, dict] = {}

    for topology in args.topologies:
        summaries, comparisons = run_topology(
            topology,
            model=model,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            m=m,
            n_range=n_range,
            num_graphs=args.num_graphs,
            seeds=args.seeds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
        )
        results_by_topology[topology] = {
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

    output = {
        "tier": "topology_ablation",
        "description": "BA vs ER vs WS at n∈[30,50] with matched average degree (~4)",
        "regime": {
            "alpha": alpha,
            "pfail": pfail,
            "budget": budget,
            "max_rounds": max_rounds,
        },
        "graph_params": {
            "n_range": list(n_range),
            "m": m,
            "avg_degree_target": 2 * m,
            "ws_k": 2 * m,
            "ws_p": 0.1,
            "er_p_formula": "2*m/n",
            "num_graphs_per_topology": args.num_graphs,
            "num_seeds": len(args.seeds),
        },
        "topologies": results_by_topology,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "topology_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved -> {summary_path}")

    write_run_metadata(
        args.output_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={"summary_path": portable_artifact_path(summary_path)},
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
