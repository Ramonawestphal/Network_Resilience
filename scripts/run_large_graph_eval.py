from __future__ import annotations

import argparse
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

DEFAULT_REFERENCE_N = 40
DEFAULT_REFERENCE_BUDGET = 2
DEFAULT_REFERENCE_MAX_ROUNDS = 20

ALPHA_VALUES_DEFAULT = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
PFAIL_VALUES_DEFAULT = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]


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
                f"{_fmt_pm(summary.final_nc)}  "
                f"{rws}  {rwf}  {label:<20}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic large-graph regime grid (BA graphs, heuristic policies only).",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100],
        help="Graph sizes n to evaluate (default: 100). Each size runs a separate grid with its own scaled budget.",
    )
    p.add_argument(
        "--num-graphs",
        type=int,
        default=10,
        help="Number of BA graphs per size (default: 10).",
    )
    p.add_argument(
        "-m",
        type=int,
        default=2,
        help="BA attachment parameter m (default: 2).",
    )
    p.add_argument(
        "--graph-base-seed",
        type=int,
        default=80_000,
        help="RNG seed for graph generation (default: 80000).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Failure seeds per graph (default: 0..9).",
    )
    p.add_argument(
        "--policies",
        nargs="+",
        choices=["random", "degree", "greedy", "betweenness", "risk"],
        default=["random", "degree", "greedy"],
        metavar="POLICY",
        help="Subset of heuristic policies (default: random degree greedy).",
    )
    p.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=ALPHA_VALUES_DEFAULT,
        help="Alpha grid for regime cells.",
    )
    p.add_argument(
        "--pfail-values",
        type=float,
        nargs="+",
        default=PFAIL_VALUES_DEFAULT,
        help="Pfail grid for regime cells.",
    )
    p.add_argument(
        "--reference-n",
        type=int,
        default=DEFAULT_REFERENCE_N,
        help="Reference n for budget / max_rounds scaling.",
    )
    p.add_argument(
        "--reference-budget",
        type=int,
        default=DEFAULT_REFERENCE_BUDGET,
        help="Reference budget before scaling to each n.",
    )
    p.add_argument(
        "--reference-max-rounds",
        type=int,
        default=DEFAULT_REFERENCE_MAX_ROUNDS,
        help="Reference max_rounds before scaling to each n.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "large_graph_eval",
        help="Output directory.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return p.parse_args()


def run_one_size(
    *,
    n: int,
    m: int,
    num_graphs: int,
    graph_base_seed: int,
    seeds: list[int],
    policies: tuple[str, ...],
    alpha_values: list[float],
    pfail_values: list[float],
    reference_n: int,
    reference_budget: int,
    reference_max_rounds: int,
    no_progress: bool,
) -> tuple[list, int, int, list[int]]:
    b_base = compute_scaled_budget(
        reference_budget, num_nodes=n, reference_n=reference_n
    )
    max_rounds = compute_scaled_max_rounds(
        reference_max_rounds, num_nodes=n, reference_n=reference_n
    )
    budgets = [b_base, b_base + 1, b_base + 2, b_base + 3]

    rng = Random(graph_base_seed + n)
    graphs = [
        make_ba_graph(n=n, m=m, seed=rng.randint(0, 10**9)) for _ in range(num_graphs)
    ]

    all_factories = build_policy_factories(base_seed=graph_base_seed)
    factories = {name: all_factories[name] for name in policies}

    num_cells = len(alpha_values) * len(pfail_values) * len(budgets)
    rollouts_per_cell = num_graphs * len(seeds) * len(factories)
    total_ticks = num_cells * rollouts_per_cell

    progress_tick = None
    bar = None
    if not no_progress:
        from tqdm import tqdm  # type: ignore[import-untyped]

        bar = tqdm(
            total=total_ticks,
            desc=f"n={n} rollouts",
            unit="ep",
            leave=True,
        )

        def make_tick(b: object):
            def _tick() -> None:
                b.update(1)  # type: ignore[attr-defined]

            return _tick

        progress_tick = make_tick(bar)

    try:
        cells = build_regime_cells(
            graphs,
            factories,
            alpha_values=alpha_values,
            pfail_values=pfail_values,
            budgets=budgets,
            max_rounds=max_rounds,
            seeds=seeds,
            scale_budget=False,
            scale_max_rounds=False,
            reference_n=reference_n,
            progress_tick=progress_tick,
        )
    finally:
        if bar is not None:
            bar.close()

    return cells, b_base, max_rounds, budgets


def main() -> None:
    args = parse_args()
    policies = tuple(dict.fromkeys(args.policies))
    sizes = list(dict.fromkeys(args.sizes))

    print(
        f"Large-graph regime evaluation: sizes={sizes}, m={args.m}, "
        f"graphs={args.num_graphs}, seeds={args.seeds}"
    )
    print(f"  policies={list(policies)}")
    print(f"  alpha_values={args.alpha_values}")
    print(f"  pfail_values={args.pfail_values}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for n in sizes:
        print(f"\n--- n={n} ---")
        cells, b_base, max_rounds, budgets = run_one_size(
            n=n,
            m=args.m,
            num_graphs=args.num_graphs,
            graph_base_seed=args.graph_base_seed,
            seeds=list(args.seeds),
            policies=policies,
            alpha_values=list(args.alpha_values),
            pfail_values=list(args.pfail_values),
            reference_n=args.reference_n,
            reference_budget=args.reference_budget,
            reference_max_rounds=args.reference_max_rounds,
            no_progress=args.no_progress,
        )
        print(f"  b_base={b_base}, budgets={budgets}, max_rounds={max_rounds}")
        print(f"  cells for this n: {len(cells)}")

        size_tag = f"n{n}"
        all_cells_path = args.output_dir / f"all_cells_{size_tag}.json"
        with all_cells_path.open("w", encoding="utf-8") as f:
            json.dump([serialize_regime_cell(c) for c in cells], f, indent=2)

        summary_path = args.output_dir / f"regime_summary_{size_tag}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summarize_regime_buckets(cells), f, indent=2)

        print(f"  Saved cells -> {all_cells_path}")
        print(f"  Saved summary -> {summary_path}")

        print_results_table(cells)
        buckets = summarize_regime_buckets(cells)
        print(f"\n  Regime bucket summary (n={n}):")
        for bucket_name, bucket_data in buckets.items():
            count = bucket_data.get("cell_count", 0)
            print(f"    {bucket_name:<20}: {count} cells")

    write_run_metadata(
        args.output_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        extra={
            "sizes": sizes,
            "m": args.m,
            "num_graphs": args.num_graphs,
            "graph_base_seed": args.graph_base_seed,
            "seeds": list(args.seeds),
            "alpha_values": list(args.alpha_values),
            "pfail_values": list(args.pfail_values),
            "reference_n": args.reference_n,
            "reference_budget": args.reference_budget,
            "reference_max_rounds": args.reference_max_rounds,
            "policies": list(policies),
        },
    )

    print(f"\nDone. Output under {args.output_dir}")


if __name__ == "__main__":
    main()
