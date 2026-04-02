from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
EVALUATE_POLICY = ROOT / "scripts" / "evaluate_policy.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deprecated compatibility wrapper around scripts/evaluate_policy.py."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--graph-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Exact graph sizes to evaluate, e.g. 100 300 500 1000.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=1,
        help="Number of graphs to sample per graph size.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Matched rollout seeds.",
    )
    parser.add_argument("--alpha", type=float, required=True, help="Cascade alpha.")
    parser.add_argument("--pfail", type=float, required=True, help="Initial failure probability.")
    parser.add_argument("--budget", type=int, required=True, help="Recovery budget.")
    parser.add_argument("--max-rounds", type=int, required=True, help="Maximum recovery rounds.")
    parser.add_argument(
        "--reference-n",
        type=int,
        default=None,
        help="Optional reference graph size for canonical budget scaling.",
    )
    parser.add_argument("--m", type=int, default=2, help="BA attachment parameter.")
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=["rl", "degree"],
        help="Subset of policies to evaluate.",
    )
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=1007,
        help="Base seed for graph generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "scaling",
        help="Directory for scaling evaluation artifacts.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="ANC threshold used for threshold_hit_fraction.",
    )
    return parser.parse_args()


def _extract_policy_summaries(cell: dict[str, object]) -> dict[str, dict[str, float]]:
    policy_summaries = cell["policy_summaries"]
    if not isinstance(policy_summaries, dict):
        raise ValueError("Unexpected grid summary format: policy_summaries must be a mapping.")

    serialized: dict[str, dict[str, float]] = {}
    for policy_name, summary in policy_summaries.items():
        if not isinstance(summary, dict):
            raise ValueError(f"Unexpected summary format for policy {policy_name!r}.")
        serialized[str(policy_name)] = {
            "final_anc_mean": float(summary["final_anc"]["mean"]),
            "final_anc_stderr": float(summary["final_anc"]["stderr"]),
            "threshold_hit_mean": float(summary["threshold_hit_fraction"]["mean"]),
            "threshold_hit_stderr": float(summary["threshold_hit_fraction"]["stderr"]),
            "rounds_mean": float(summary["rounds"]["mean"]),
            "rounds_stderr": float(summary["rounds"]["stderr"]),
            "solved_fraction_mean": float(summary["solved_fraction"]["mean"]),
            "solved_fraction_stderr": float(summary["solved_fraction"]["stderr"]),
        }
    return serialized


def main() -> None:
    args = parse_args()
    print(
        "Deprecated: use scripts/evaluate_policy.py directly for maintained scaling runs.",
        flush=True,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    size_results: list[dict[str, object]] = []
    env_metadata: dict[str, object] | None = None
    tau: float | None = None

    for graph_size in args.graph_sizes:
        temp_output_dir = output_dir / f".compat_n{graph_size}"
        command = [
            sys.executable,
            str(EVALUATE_POLICY),
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--alpha-values",
            str(args.alpha),
            "--pfail-values",
            str(args.pfail),
            "--budgets",
            str(args.budget),
            "--num-graphs",
            str(args.num_graphs),
            "--max-rounds",
            str(args.max_rounds),
            "--n-range",
            str(graph_size),
            str(graph_size),
            "--graph-seed",
            str(args.graph_seed + graph_size),
            "--output-dir",
            str(temp_output_dir),
            "--scale-budget",
            "--policies",
            *args.policies,
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        if args.reference_n is not None:
            command.extend(["--reference-n", str(args.reference_n)])
        if args.tau is not None:
            command.extend(["--tau", str(args.tau)])

        size_started = perf_counter()
        subprocess.run(command, check=True)
        elapsed_seconds = perf_counter() - size_started

        grid_summary_path = temp_output_dir / "evaluation_grid_summary.json"
        with grid_summary_path.open("r", encoding="utf-8") as file:
            grid_summary = json.load(file)

        cells = grid_summary.get("cells", [])
        if len(cells) != 1:
            raise ValueError(
                "evaluate_scaling expected exactly one regime cell per graph size "
                f"(from evaluate_policy subprocess); got cells_len={len(cells)}. "
                f"Requested alpha={args.alpha!r} pfail={args.pfail!r} budget={args.budget!r} "
                f"max_rounds={args.max_rounds!r} graph_size={graph_size!r}. "
                f"grid_summary keys={sorted(grid_summary.keys())!r}. "
                "If multiple cells appear, adjust CLI filters or evaluate_policy grid layout."
            )
        cell = cells[0]
        env_metadata = dict(grid_summary["env"])
        tau = float(grid_summary["tau"])
        size_results.append(
            {
                "graph_size": graph_size,
                "num_graphs": args.num_graphs,
                "seeds": list(args.seeds),
                "elapsed_seconds": elapsed_seconds,
                "policy_summaries": _extract_policy_summaries(cell),
            }
        )
        policy_text = "  ".join(
            f"{name}: final_anc={summary['final_anc_mean']:.3f}"
            for name, summary in size_results[-1]["policy_summaries"].items()
        )
        print(f"[scaling] n={graph_size} elapsed={elapsed_seconds:.1f}s {policy_text}", flush=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)

    output_path = output_dir / (
        f"{args.checkpoint.parent.name}_{'_'.join(dict.fromkeys(args.policies))}_"
        f"a{args.alpha:.2f}_p{args.pfail:.2f}_b{args.budget}_mr{args.max_rounds}.json"
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "alpha": args.alpha,
        "pfail": args.pfail,
        "budget": args.budget,
        "max_rounds": args.max_rounds,
        "m": args.m,
        "graph_seed": args.graph_seed,
        "tau": tau,
        "policies": list(dict.fromkeys(args.policies)),
        "env": env_metadata,
        "elapsed_seconds_total": perf_counter() - started,
        "results": size_results,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved scaling summary to {output_path}")


if __name__ == "__main__":
    main()
