from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.evaluation import (
    build_policy_factories,
    build_regime_cells,
    serialize_regime_cell,
    summarize_regime_buckets,
)
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.reproducibility import write_run_metadata

NUM_GRAPHS = 10
N = 100
M = 2
GRAPH_BASE_SEED = 80_000
SEEDS = list(range(10))
ALPHA_VALUES = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
PFAIL_VALUES = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
REFERENCE_N = 40
REFERENCE_BUDGET = 2
REFERENCE_MAX_ROUNDS = 20
POLICIES = ("random", "degree", "greedy")
OUTPUT_DIR = ROOT / "experiments" / "large_graph_eval"


def _fmt(metric, *, width: int = 5) -> str:
    if metric is None:
        return f"{'—':>{width}}"
    return f"{metric.mean:>{width}.2f}"


def _fmt_pm(metric) -> str:
    if metric is None:
        return f"{'—':>12}"
    return f"{metric.mean:>5.3f}±{metric.stderr:.3f}"


def print_results_table(cells: list) -> None:
    header = (
        f"{'alpha':>5} {'pfail':>5} {'budget':>6} {'policy':>10}  "
        f"{'solved':>12}  {'final_anc':>12}  {'rws':>5}  {'rwf':>5}  {'regime':<20}"
    )
    print()
    print(header)
    print("-" * len(header))
    for cell in cells:
        label = cell.diagnostics.regime_label
        for policy_name, summary in sorted(cell.policy_summaries.items()):
            rws = _fmt(summary.rounds_when_solved)
            rwf = _fmt(summary.rounds_when_failed)
            print(
                f"{cell.alpha:>5.2f} {cell.pfail:>5.2f} {cell.budget:>6d} {policy_name:>10}  "
                f"{_fmt_pm(summary.solved_fraction)}  "
                f"{_fmt_pm(summary.final_anc)}  "
                f"{rws}  {rwf}  {label:<20}"
            )


def main() -> None:
    # Pre-compute scaled parameters for n=100
    b_base = compute_scaled_budget(REFERENCE_BUDGET, num_nodes=N, reference_n=REFERENCE_N)
    max_rounds = compute_scaled_max_rounds(REFERENCE_MAX_ROUNDS, num_nodes=N, reference_n=REFERENCE_N)
    budgets = [b_base, b_base + 1, b_base + 2, b_base + 3]

    print(f"Large-graph regime evaluation: n={N}, m={M}, graphs={NUM_GRAPHS}, seeds={SEEDS}")
    print(f"  b_base={b_base}, budgets={budgets}, max_rounds={max_rounds}")
    print(f"  alpha_values={ALPHA_VALUES}")
    print(f"  pfail_values={PFAIL_VALUES}")
    print(f"  policies={list(POLICIES)}")
    print(f"  total cells: {len(ALPHA_VALUES)}×{len(PFAIL_VALUES)}×{len(budgets)} = "
          f"{len(ALPHA_VALUES) * len(PFAIL_VALUES) * len(budgets)}")

    # Generate fixed BA graphs
    rng = Random(GRAPH_BASE_SEED)
    graphs = [make_ba_graph(n=N, m=M, seed=rng.randint(0, 10**9)) for _ in range(NUM_GRAPHS)]

    # Build subset of policy factories
    all_factories = build_policy_factories(base_seed=GRAPH_BASE_SEED)
    factories = {name: all_factories[name] for name in POLICIES}

    # Run full regime grid
    print("\nRunning regime grid (this may take several minutes)...")
    cells = build_regime_cells(
        graphs,
        factories,
        alpha_values=ALPHA_VALUES,
        pfail_values=PFAIL_VALUES,
        budgets=budgets,
        max_rounds=max_rounds,
        seeds=SEEDS,
        scale_budget=False,
        scale_max_rounds=False,
        reference_n=REFERENCE_N,
    )
    print(f"Done. {len(cells)} cells evaluated.")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cells_path = OUTPUT_DIR / "all_cells.json"
    with all_cells_path.open("w", encoding="utf-8") as f:
        json.dump([serialize_regime_cell(c) for c in cells], f, indent=2)

    summary_path = OUTPUT_DIR / "regime_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summarize_regime_buckets(cells), f, indent=2)

    write_run_metadata(
        OUTPUT_DIR / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        extra={
            "n": N,
            "m": M,
            "num_graphs": NUM_GRAPHS,
            "graph_base_seed": GRAPH_BASE_SEED,
            "seeds": SEEDS,
            "alpha_values": ALPHA_VALUES,
            "pfail_values": PFAIL_VALUES,
            "b_base": b_base,
            "budgets": budgets,
            "max_rounds": max_rounds,
            "reference_n": REFERENCE_N,
            "policies": list(POLICIES),
        },
    )

    # Print human-readable table
    print_results_table(cells)

    # Regime bucket summary
    buckets = summarize_regime_buckets(cells)
    print("\nRegime bucket summary:")
    for bucket_name, bucket_data in buckets.items():
        count = bucket_data.get("cell_count", 0)
        print(f"  {bucket_name:<20}: {count} cells")

    print(f"\nSaved {len(cells)} cells to {all_cells_path}")
    print(f"Saved regime summary to {summary_path}")


if __name__ == "__main__":
    main()
