from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import DEFAULT_REFERENCE_N
from cascading_rl.evaluation import build_policy_factories, estimate_minimum_budget
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network
from scripts.reproducibility import write_run_metadata

SUPPORTED_POLICIES = ("rl", "random", "degree", "risk", "greedy", "betweenness")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def resolve_env_kwargs(config: dict[str, Any]) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimum-budget search for configured policies.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "learner" / "recovery_q.pt",
        help="Path to the trained checkpoint used for the RL policy.",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--pfail", type=float, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--num-graphs", type=int, default=1)
    parser.add_argument("--graph-seed", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--policies", type=str, nargs="+", default=None)
    parser.add_argument("--scale-budget", action="store_true")
    parser.add_argument("--reference-n", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "reference_regime",
        help="Directory for budget-search artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    regime = training["regime"]
    graph_cfg = training["graph"]
    evaluation = config["evaluation"]
    budget_scaling = config.get("budget_scaling", {})
    env_kwargs = resolve_env_kwargs(config)

    alpha = float(args.alpha if args.alpha is not None else regime["alpha"])
    pfail = float(args.pfail if args.pfail is not None else regime["pfail"])
    max_rounds = int(args.max_rounds if args.max_rounds is not None else regime["max_rounds"])
    budgets = list(args.budgets) if args.budgets is not None else [int(value) for value in evaluation["budgets"]]
    tau = float(args.tau if args.tau is not None else evaluation["tau"])
    scale_budget = bool(args.scale_budget or budget_scaling.get("enabled", False))
    reference_n = int(
        args.reference_n
        if args.reference_n is not None
        else budget_scaling.get("reference_n", DEFAULT_REFERENCE_N)
    )
    graph_seed = int(args.graph_seed if args.graph_seed is not None else training["seed"] + 1500)
    selected_policies = list(dict.fromkeys(args.policies or list(SUPPORTED_POLICIES)))

    invalid = [policy for policy in selected_policies if policy not in SUPPORTED_POLICIES]
    if invalid:
        raise ValueError(
            f"Unsupported policies: {invalid}. Supported values: {list(SUPPORTED_POLICIES)}"
        )

    graphs = make_graph_batch(
        num_graphs=args.num_graphs,
        n_range=tuple(graph_cfg["n_range"]),
        m=int(graph_cfg["m"]),
        seed=graph_seed,
    )
    representative_graph = graphs[0]

    base_factories = build_policy_factories(base_seed=graph_seed)
    policy_map: dict[str, Any] = {}
    for policy_name in selected_policies:
        if policy_name == "rl":
            model, _ = load_q_network(args.checkpoint)
            policy_map["rl"] = build_greedy_policy(model, batch_actions=True)
        else:
            policy_map[policy_name] = base_factories[policy_name](0, 0)

    results: dict[str, Any] = {}
    for policy_name, policy in policy_map.items():
        minimum_budget, search_curve = estimate_minimum_budget(
            representative_graph,
            policy,
            tau=tau,
            budgets=budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=alpha,
            pfail=pfail,
            max_rounds=max_rounds,
            env_kwargs=env_kwargs,
            scale_budget=scale_budget,
            reference_n=reference_n,
        )
        results[policy_name] = {
            "minimum_budget": minimum_budget,
            "search_curve": {
                str(budget): {"final_anc_mean": mean, "final_anc_stderr": stderr}
                for budget, (mean, stderr) in search_curve.items()
            },
        }

    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "alpha": alpha,
        "pfail": pfail,
        "max_rounds": max_rounds,
        "tau": tau,
        "budgets": budgets,
        "graph_seed": graph_seed,
        "num_graphs": args.num_graphs,
        "env": env_kwargs,
        "scaling": {
            "scale_budget": scale_budget,
            "reference_n": reference_n,
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "budget_search_summary.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={
            "output_dir": str(args.output_dir),
            "env": env_kwargs,
            "scaling": {
                "scale_budget": scale_budget,
                "reference_n": reference_n,
            },
        },
    )

    print(f"Saved budget-search summary to {output_path}")
    for policy_name, result in results.items():
        print(f"{policy_name}: minimum_budget={result['minimum_budget']}")


if __name__ == "__main__":
    main()
