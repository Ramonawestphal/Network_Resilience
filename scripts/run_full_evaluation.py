"""Run the complete evaluation suite across all parameter combinations.

Six tiers, all driven by the same (alpha, pfail, budget) grid:

  Tier 1a — BA 30-50 in-dist param grid  (evaluate_param_generalization.py)
             Training-distribution graph size and topology. One invocation.
             Output: experiments/eval_param_generalization/ba_30_50/

  Tier 1b — Topology ablation            (evaluate_topology_ablation.py)
             BA vs ER vs WS at n∈[30,50]. One invocation per cell.
             Output: experiments/eval_topology_ablation/a{alpha}_p{pfail}_b{budget}/

  Tier 1c — Large BA param grid          (evaluate_param_generalization.py)
             BA n∈[100,300] — tests size generalisation. One invocation.
             Scaled down: 30 graphs, 5 seeds (vs 100/10 for small graphs).
             Output: experiments/eval_param_generalization/ba_100_300/

  Tier 1d — ER param grid                (evaluate_param_generalization.py)
             Erdős-Rényi n∈[30,50]. One invocation.
             Output: experiments/eval_param_generalization/er_30_50/

  Tier 1e — WS param grid                (evaluate_param_generalization.py)
             Watts-Strogatz n∈[30,50]. One invocation.
             Output: experiments/eval_param_generalization/ws_30_50/

  Tier 2  — OOD real-world               (evaluate_real_world.py)
             IEEE 300-bus. One invocation per cell.
             Output: experiments/eval_real_world/a{alpha}_p{pfail}_b{budget}/

Grid (matches evaluate_param_generalization.py defaults)
---------------------------------------------------------
  alpha  : [0.10, 0.20, 0.25, 0.30]  -- low / mid / train / high capacity slack
  pfail  : [0.05, 0.15, 0.20, 0.25]  -- low / mid / train / high failure rate
  budget : [1, 2, 3]                 -- budget=2 is training budget
  Total  : 48 cells  (training cell: alpha=0.25, pfail=0.20, budget=2)

Usage
-----
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --alpha 0.20 0.25 --pfail 0.15 0.20 --budget 1 2
    python scripts/run_full_evaluation.py --skip-indist    # skip Tier 1a
    python scripts/run_full_evaluation.py --skip-large-ba  # skip Tier 1c
    python scripts/run_full_evaluation.py --skip-er        # skip Tier 1d
    python scripts/run_full_evaluation.py --skip-ws        # skip Tier 1e
    python scripts/run_full_evaluation.py --skip-topo      # skip Tier 1b
    python scripts/run_full_evaluation.py --skip-ood       # skip Tier 2
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ALPHA  = [0.10, 0.20, 0.25, 0.30]   # 0.25 = training alpha
DEFAULT_PFAIL  = [0.05, 0.15, 0.20, 0.25]   # 0.20 = training pfail
DEFAULT_BUDGET = [1, 2, 3]                  # budget=2 = training budget


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full evaluation suite across parameter grid.")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "experiments" / "learner" / "recovery_q.pt")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    p.add_argument("--alpha",  type=float, nargs="+", default=DEFAULT_ALPHA)
    p.add_argument("--pfail",  type=float, nargs="+", default=DEFAULT_PFAIL)
    p.add_argument("--budget", type=int,   nargs="+", default=DEFAULT_BUDGET)
    p.add_argument("--num-graphs", type=int, default=40,
                   help="Graphs per cell for Tiers 1a/1b/1d/1e (default: 40).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)),
                   help="Failure seeds for in-dist and topo ablation (default: 0..4).")
    p.add_argument("--ood-seeds", type=int, nargs="+", default=list(range(5)),
                   help="Failure seeds for OOD real-world (default: 0..4).")
    # Large-BA specific — scaled down to keep runtime manageable
    p.add_argument("--large-ba-num-graphs", type=int, default=30,
                   help="Graphs for Tier 1c large-BA sweep (default: 30).")
    p.add_argument("--large-ba-seeds", type=int, nargs="+", default=list(range(5)),
                   help="Failure seeds for Tier 1c (default: 0..4).")
    p.add_argument("--large-ba-n-low", type=int, default=100,
                   help="Min graph size for Tier 1c (default: 100).")
    p.add_argument("--large-ba-n-high", type=int, default=300,
                   help="Max graph size for Tier 1c (default: 300).")
    p.add_argument("--skip-indist",   action="store_true", help="Skip Tier 1a  (BA 30-50 in-dist).")
    p.add_argument("--skip-large-ba", action="store_true", help="Skip Tier 1c  (BA 100-500).")
    p.add_argument("--skip-er",       action="store_true", help="Skip Tier 1d  (ER 30-50).")
    p.add_argument("--skip-ws",       action="store_true", help="Skip Tier 1e  (WS 30-50).")
    p.add_argument("--skip-topo",     action="store_true", help="Skip Tier 1b  (topology ablation).")
    p.add_argument("--skip-ood",      action="store_true", help="Skip Tier 2   (OOD real-world).")
    return p.parse_args()


def _cell_tag(alpha: float, pfail: float, budget: int) -> str:
    return f"a{alpha}_p{pfail}_b{budget}"


def run(cmd: list) -> None:
    """Run a subprocess command, streaming output, exit on failure."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\nERROR: command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()
    grid = list(itertools.product(args.alpha, args.pfail, args.budget))
    total_cells = len(grid)
    seeds_str     = [str(s) for s in args.seeds]
    ood_seeds_str = [str(s) for s in args.ood_seeds]
    alpha_str  = [str(a) for a in args.alpha]
    pfail_str  = [str(p) for p in args.pfail]
    budget_str = [str(b) for b in args.budget]

    print(f"Grid: {len(args.alpha)} alpha x {len(args.pfail)} pfail x "
          f"{len(args.budget)} budget = {total_cells} cells")
    print(f"Checkpoint: {args.checkpoint}")

    # ------------------------------------------------------------------
    # Tier 1a: in-distribution param grid
    # evaluate_param_generalization.py accepts full lists and iterates
    # over all (alpha, pfail, budget) cells internally — one invocation
    # covers all 100 cells.
    # ------------------------------------------------------------------
    if not args.skip_indist:
        print(f"\n{'='*60}")
        print(f"TIER 1a — In-distribution param grid: BA n∈[30,50] ({total_cells} cells)")
        print(f"  alpha  = {args.alpha}")
        print(f"  pfail  = {args.pfail}")
        print(f"  budget = {args.budget}")
        print(f"{'='*60}")
        run([
            "scripts/evaluate_param_generalization.py",
            "--checkpoint", args.checkpoint,
            "--config",     args.config,
            "--graph-type", "ba",
            "--alpha",      *alpha_str,
            "--pfail",      *pfail_str,
            "--budget",     *budget_str,
            "--num-graphs", str(args.num_graphs),
            "--seeds",      *seeds_str,
            "--output-dir", ROOT / "experiments" / "eval_param_generalization" / "ba_30_50",
        ])

    # ------------------------------------------------------------------
    # Tier 1c: param sweep on large BA graphs (n∈[100,500])
    # ------------------------------------------------------------------
    if not args.skip_large_ba:
        large_ba_seeds_str = [str(s) for s in args.large_ba_seeds]
        n_range_tag = f"{args.large_ba_n_low}_{args.large_ba_n_high}"
        print(f"\n{'='*60}")
        print(f"TIER 1c — Large BA param grid: BA n∈[{args.large_ba_n_low},{args.large_ba_n_high}] "
              f"({total_cells} cells, {args.large_ba_num_graphs} graphs, {len(args.large_ba_seeds)} seeds)")
        print(f"{'='*60}")
        run([
            "scripts/evaluate_param_generalization.py",
            "--checkpoint", args.checkpoint,
            "--config",     args.config,
            "--graph-type", "ba",
            "--n-low",      str(args.large_ba_n_low),
            "--n-high",     str(args.large_ba_n_high),
            "--alpha",      *alpha_str,
            "--pfail",      *pfail_str,
            "--budget",     *budget_str,
            "--num-graphs", str(args.large_ba_num_graphs),
            "--seeds",      *large_ba_seeds_str,
            "--output-dir", ROOT / "experiments" / "eval_param_generalization" / f"ba_{n_range_tag}",
            "--sequential-greedy",
        ])

    # ------------------------------------------------------------------
    # Tier 1d: param sweep on ER graphs (n∈[30,50])
    # ------------------------------------------------------------------
    if not args.skip_er:
        print(f"\n{'='*60}")
        print(f"TIER 1d — ER param grid: ER n∈[30,50] ({total_cells} cells)")
        print(f"{'='*60}")
        run([
            "scripts/evaluate_param_generalization.py",
            "--checkpoint", args.checkpoint,
            "--config",     args.config,
            "--graph-type", "er",
            "--alpha",      *alpha_str,
            "--pfail",      *pfail_str,
            "--budget",     *budget_str,
            "--num-graphs", str(args.num_graphs),
            "--seeds",      *seeds_str,
            "--output-dir", ROOT / "experiments" / "eval_param_generalization" / "er_30_50",
        ])

    # ------------------------------------------------------------------
    # Tier 1e: param sweep on WS graphs (n∈[30,50])
    # ------------------------------------------------------------------
    if not args.skip_ws:
        print(f"\n{'='*60}")
        print(f"TIER 1e — WS param grid: WS n∈[30,50] ({total_cells} cells)")
        print(f"{'='*60}")
        run([
            "scripts/evaluate_param_generalization.py",
            "--checkpoint", args.checkpoint,
            "--config",     args.config,
            "--graph-type", "ws",
            "--alpha",      *alpha_str,
            "--pfail",      *pfail_str,
            "--budget",     *budget_str,
            "--num-graphs", str(args.num_graphs),
            "--seeds",      *seeds_str,
            "--output-dir", ROOT / "experiments" / "eval_param_generalization" / "ws_30_50",
        ])

    # ------------------------------------------------------------------
    # Tier 1b: topology ablation — BA vs ER vs WS
    # Each script call handles one (alpha, pfail, budget) cell.
    # ------------------------------------------------------------------
    if not args.skip_topo:
        print(f"\n{'='*60}")
        print(f"TIER 1b — Topology ablation ({total_cells} cells)")
        print(f"{'='*60}")
        for idx, (alpha, pfail, budget) in enumerate(grid, 1):
            tag = _cell_tag(alpha, pfail, budget)
            out = ROOT / "experiments" / "eval_topology_ablation" / tag
            print(f"\n[{idx}/{total_cells}] {tag}")
            run([
                "scripts/evaluate_topology_ablation.py",
                "--checkpoint", args.checkpoint,
                "--config",     args.config,
                "--alpha",      str(alpha),
                "--pfail",      str(pfail),
                "--budget",     str(budget),
                "--num-graphs", str(args.num_graphs),
                "--seeds",      *seeds_str,
                "--output-dir", out,
            ])

    # ------------------------------------------------------------------
    # Tier 2: OOD real-world (IEEE 300-bus)
    # Each script call handles one (alpha, pfail, budget) cell.
    # ------------------------------------------------------------------
    if not args.skip_ood:
        print(f"\n{'='*60}")
        print(f"TIER 2 — OOD real-world / IEEE 300-bus ({total_cells} cells)")
        print(f"{'='*60}")
        for idx, (alpha, pfail, budget) in enumerate(grid, 1):
            tag = _cell_tag(alpha, pfail, budget)
            out = ROOT / "experiments" / "eval_real_world" / tag
            print(f"\n[{idx}/{total_cells}] {tag}")
            run([
                "scripts/evaluate_real_world.py",
                "--checkpoint", args.checkpoint,
                "--config",     args.config,
                "--alpha",      str(alpha),
                "--pfail",      str(pfail),
                "--budget",     str(budget),
                "--seeds",      *ood_seeds_str,
                "--output-dir", out,
            ])

    # ------------------------------------------------------------------
    # Plots: generate one figure set per (alpha, pfail, budget) cell
    # where both topo ablation and OOD results are available.
    # ------------------------------------------------------------------
    if not args.skip_topo and not args.skip_ood:
        print(f"\n{'='*60}")
        print("PLOTS — topology ablation + OOD per cell")
        print(f"{'='*60}")
        for alpha, pfail, budget in grid:
            tag = _cell_tag(alpha, pfail, budget)
            topo_json = ROOT / "experiments" / "eval_topology_ablation" / tag / "topology_ablation_summary.json"
            ood_json  = ROOT / "experiments" / "eval_real_world" / tag / "ieee300" / "evaluation_summary.json"
            out       = ROOT / "experiments" / "eval_plots" / tag
            if not topo_json.exists() or not ood_json.exists():
                print(f"  SKIP {tag}: missing result files")
                continue
            run([
                "scripts/plot_evaluation_tiers.py",
                "--topo-json", topo_json,
                "--ood-json",  ood_json,
                "--out-dir",   out,
                "--tag",       tag,
            ])

    print(f"\n{'='*60}")
    print("All evaluations complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
