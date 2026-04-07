from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import DEFAULT_REFERENCE_N, compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation import (
    RegimeCellResult,
    build_policy_factories,
    build_regime_cells,
    compute_regime_diagnostics,
    estimate_minimum_budget,
    evaluate_policy_factories_on_graphs,
    rollout_policy,
    serialize_regime_cell,
    summarize_episode_results,
    summarize_regime_buckets,
)
from cascading_rl.evaluation.benchmarks import final_nc_failure_threshold_for_reporting
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network
from cascading_rl.reproducibility import portable_artifact_path, write_run_metadata

SUPPORTED_POLICIES = ("rl", "random", "degree", "risk", "greedy", "betweenness")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained recovery learner.")
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
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--grid-source",
        choices=("training", "regime_mapping"),
        default="training",
        help="Which config section should define the regime grid for robust evaluation.",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional alpha override for the grid evaluation.",
    )
    parser.add_argument(
        "--pfail-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional pfail override for the grid evaluation.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Optional budget override for the grid evaluation.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=None,
        help="Optional graph-count override for the grid evaluation.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional matched-seed override for the grid evaluation.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional max-rounds override for the grid evaluation.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of policies to evaluate.",
    )
    parser.add_argument(
        "--n-range",
        type=int,
        nargs=2,
        metavar=("MIN_N", "MAX_N"),
        default=None,
        help="Optional graph-size range override for generated evaluation graphs.",
    )
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=None,
        help="Optional graph-generation seed override for evaluation graphs.",
    )
    parser.add_argument(
        "--scale-budget",
        action="store_true",
        help="Scale the evaluation budget with graph size using beta = budget / reference_n.",
    )
    parser.add_argument(
        "--scale-max-rounds",
        action="store_true",
        help=(
            "Scale max_rounds linearly with graph size (same reference_n as budget scaling: "
            "round(max_rounds * n / reference_n))."
        ),
    )
    parser.add_argument(
        "--reference-n",
        type=int,
        default=None,
        help=(
            "Reference graph size used to compute the recovery rate beta = budget / "
            "reference_n when budget scaling is active. Defaults to config.budget_scaling.reference_n "
            f"or {DEFAULT_REFERENCE_N} when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation artifacts (default: training.benchmark_dir from config).",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help="Load fixed eval instances from a pickle file (e.g. eval_sets/ds_validation.pkl).",
    )
    parser.add_argument(
        "--eval-set-log",
        type=Path,
        default=None,
        help=(
            "With --eval-set only: write the same report lines to this UTF-8 file "
            "(directories are created; still prints to stdout)."
        ),
    )
    return parser.parse_args()


def serialize_path(path: Path) -> str:
    return portable_artifact_path(path)


def build_eval_policy_factories(
    checkpoint_path: Path,
    *,
    base_seed: int,
    selected_policies: list[str] | None = None,
) -> dict[str, Any]:
    selected_policies = list(dict.fromkeys(selected_policies or list(SUPPORTED_POLICIES)))
    invalid = [policy for policy in selected_policies if policy not in SUPPORTED_POLICIES]
    if invalid:
        raise ValueError(
            f"Unsupported policies: {invalid}. Supported values: {list(SUPPORTED_POLICIES)}"
        )

    rl_policy = None
    if "rl" in selected_policies:
        model, _ = load_q_network(checkpoint_path)
        rl_policy = build_greedy_policy(model, batch_actions=True)
    base_factories = build_policy_factories(base_seed=base_seed)
    policy_factories: dict[str, Any] = {}
    for policy_name in selected_policies:
        if policy_name == "rl":
            if rl_policy is None:
                raise ValueError("RL policy requested but no checkpoint could be loaded.")
            policy_factories["rl"] = lambda _graph_index, _seed: rl_policy
        else:
            policy_factories[policy_name] = base_factories[policy_name]
    return policy_factories


def resolve_env_kwargs(config: dict[str, Any]) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    abandon_raw = regime.get("abandonment_nc_threshold")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
        "abandonment_nc_threshold": (
            float(abandon_raw) if abandon_raw is not None else None
        ),
    }


def resolve_grid_spec(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config["training"]
    evaluation = config["evaluation"]
    regime_mapping = config["regime_mapping"]

    if args.grid_source == "training":
        regime = training["regime"]
        graph_cfg = training["graph"]
        alpha_values = regime.get("alpha_values") or [regime["alpha"]]
        pfail_values = regime.get("pfail_values") or [regime["pfail"]]
        budgets = evaluation.get("budgets") or [regime["budget"]]
        num_graphs = int(training["benchmark_graphs"])
        seeds = list(training["benchmark_seeds"])
        max_rounds = int(regime["max_rounds"])
        graph_seed = int(training["seed"]) + 1000
        n_range = tuple(graph_cfg["n_range"])
        m = int(graph_cfg["m"])
    elif args.grid_source == "regime_mapping":
        graph_cfg = config["graph"]
        alpha_values = list(regime_mapping["alpha_values"])
        pfail_values = list(regime_mapping["pfail_values"])
        budgets = list(regime_mapping["budgets"])
        num_graphs = int(regime_mapping["num_graphs"])
        seeds = list(regime_mapping["seeds"])
        max_rounds = int(regime_mapping.get("max_rounds"))
        graph_seed = int(regime_mapping["graph_seed"])
        n_range = tuple(graph_cfg["n_range"])
        m = int(graph_cfg["m"])
    else:
        raise ValueError(
            f"Unknown grid_source {args.grid_source!r}; expected 'training' or 'regime_mapping'."
        )

    n_range_override = getattr(args, "n_range", None)
    resolved = {
        "alpha_values": list(args.alpha_values) if args.alpha_values is not None else list(alpha_values),
        "pfail_values": list(args.pfail_values) if args.pfail_values is not None else list(pfail_values),
        "budgets": list(args.budgets) if args.budgets is not None else list(budgets),
        "num_graphs": int(args.num_graphs) if args.num_graphs is not None else int(num_graphs),
        "seeds": list(args.seeds) if args.seeds is not None else list(seeds),
        "max_rounds": int(args.max_rounds) if args.max_rounds is not None else int(max_rounds),
        "graph_seed": int(args.graph_seed) if args.graph_seed is not None else int(graph_seed),
        "n_range": tuple(n_range_override) if n_range_override is not None else tuple(n_range),
        "m": int(m),
    }
    resolved["primary_alpha"] = float(resolved["alpha_values"][0])
    resolved["primary_pfail"] = float(resolved["pfail_values"][0])
    resolved["primary_budget"] = int(resolved["budgets"][0])
    resolved["primary_max_rounds"] = int(resolved["max_rounds"])
    return resolved


def serialize_policy_summary(summary: Any, b_star: int | None = None) -> dict[str, Any]:
    """Serialize a PolicyEvaluationSummary to a JSON-compatible dict."""
    def agg(m: Any) -> dict[str, float]:
        return {"mean": m.mean, "stderr": m.stderr}

    out: dict[str, Any] = {
        "final_nc": agg(summary.final_nc),
        "anc_fixed": agg(summary.anc_fixed),
        "anc_adaptive": agg(summary.anc_adaptive),
        "rounds": agg(summary.rounds),
        "steps": agg(summary.steps),
        "solved_fraction": agg(summary.solved_fraction),
        "rounds_when_solved": agg(summary.rounds_when_solved) if summary.rounds_when_solved is not None else None,
        "fully_restored_count": summary.fully_restored_count,
        "episode_count": summary.episode_count,
        "unsolved_low_final_nc_count": summary.unsolved_low_final_nc_count,
        "unsolved_low_final_nc_fraction": summary.unsolved_low_final_nc_fraction,
        "mean_degree_ratio": agg(summary.mean_degree_ratio),
        "mean_overload_risk": agg(summary.mean_overload_risk),
        "mean_nc_gain": agg(summary.mean_nc_gain),
        "mean_greedy_nc_gain": agg(summary.mean_greedy_nc_gain),
        "mean_action_rank": agg(summary.mean_action_rank),
        "b_star": b_star,
    }
    return out


def estimate_b_star_for_policies(
    *,
    policy_names: list[str],
    representative_graph: Any,
    rl_policy: Any | None,
    base_seed: int,
    target_solved_fraction: float,
    evaluation_budgets: list[int],
    trials: int,
    alpha: float,
    pfail: float,
    max_rounds: int | None,
    env_kwargs: dict[str, object],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
) -> dict[str, int | None]:
    base_factories = build_policy_factories(base_seed=base_seed)
    b_star: dict[str, int | None] = {}
    for policy_name in policy_names:
        if policy_name == "rl":
            if rl_policy is None:
                raise ValueError("Cannot estimate RL b_star without an RL policy.")
            policy = rl_policy
        else:
            policy = base_factories[policy_name](0, 0)
        b_star[policy_name] = estimate_minimum_budget(
            representative_graph,
            policy,
            target_solved_fraction=target_solved_fraction,
            budgets=evaluation_budgets,
            trials=trials,
            alpha=alpha,
            pfail=pfail,
            max_rounds=max_rounds,
            env_kwargs=env_kwargs,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
        )[0]
    return b_star

def build_scaling_metadata(
    *,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> dict[str, Any] | None:
    if not scale_budget and not scale_max_rounds:
        return None
    return {
        "scale_budget": scale_budget,
        "scale_max_rounds": scale_max_rounds,
        "reference_budget": reference_budget,
        "reference_n": reference_n,
        "beta": (reference_budget / reference_n) if scale_budget else None,
    }


def resolve_budget_and_rounds(
    *,
    num_nodes: int,
    pfail: float,
    budget: int,
    max_rounds: int | None,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> tuple[int, int | None]:
    resolved_budget = (
        compute_scaled_budget(
            num_nodes=num_nodes,
            reference_budget=reference_budget,
            reference_n=reference_n,
        )
        if scale_budget
        else budget
    )
    resolved_max_rounds = (
        compute_scaled_max_rounds(
            max_rounds,
            num_nodes=num_nodes,
            reference_n=reference_n,
            enabled=True,
        )
        if scale_max_rounds and max_rounds is not None
        else max_rounds
    )
    return resolved_budget, resolved_max_rounds


def log_scaling_decisions(
    *,
    graphs: list[Any],
    pfail: float,
    budget: int,
    max_rounds: int | None,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> None:
    if not scale_budget and not scale_max_rounds:
        return

    if scale_budget:
        beta = reference_budget / reference_n
        print(
            f"Budget scaling active: beta={beta:.3f} "
            f"(budget={reference_budget}, reference_n={reference_n})"
        )
    else:
        print(f"Max-round scaling active: using fixed budget={budget}")

    for num_nodes in sorted({graph.number_of_nodes() for graph in graphs}):
        scaled_budget, scaled_max_rounds = resolve_budget_and_rounds(
            num_nodes=num_nodes,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_budget=reference_budget,
            reference_n=reference_n,
        )
        print(
            f"For n={num_nodes}: scaled_budget={scaled_budget}, "
            f"scaled_max_rounds={scaled_max_rounds}"
        )


def evaluate_policy_factories_with_optional_scaling(
    graphs: list[Any],
    policy_factories: dict[str, Any],
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int | None,
    seeds: list[int],
    env_kwargs: dict[str, object],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> dict[str, Any]:
    if not scale_budget and not scale_max_rounds:
        return evaluate_policy_factories_on_graphs(
            graphs,
            policy_factories,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            seeds=seeds,
            env_kwargs=env_kwargs,
            reference_n=reference_n,
        )

    episode_results_by_policy: dict[str, list[Any]] = {name: [] for name in policy_factories}
    for graph_index, graph in enumerate(graphs):
        resolved_budget, resolved_max_rounds = resolve_budget_and_rounds(
            num_nodes=graph.number_of_nodes(),
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_budget=reference_budget,
            reference_n=reference_n,
        )
        for seed in seeds:
            for policy_name, policy_factory in policy_factories.items():
                env = RecoveryEnv(
                    graph,
                    alpha=alpha,
                    pfail=pfail,
                    budget=resolved_budget,
                    max_rounds=resolved_max_rounds,
                    seed=seed,
                    **env_kwargs,
                )
                policy = policy_factory(graph_index, seed)
                result = rollout_policy(env, policy, seed=seed)
                episode_results_by_policy[policy_name].append(result)

    thr = final_nc_failure_threshold_for_reporting(env_kwargs)
    return {
        policy_name: summarize_episode_results(
            episode_results,
            final_nc_failure_threshold=thr,
        )
        for policy_name, episode_results in episode_results_by_policy.items()
    }


def build_regime_cells_with_optional_scaling(
    graphs: list[Any],
    policy_factories: dict[str, Any],
    *,
    alpha_values: list[float],
    pfail_values: list[float],
    budgets: list[int],
    max_rounds: int | None,
    seeds: list[int],
    hopeless_threshold: float,
    trivial_threshold: float,
    spread_threshold: float,
    env_kwargs: dict[str, object],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> list[RegimeCellResult]:
    if not scale_budget and not scale_max_rounds:
        return build_regime_cells(
            graphs,
            policy_factories,
            alpha_values=alpha_values,
            pfail_values=pfail_values,
            budgets=budgets,
            max_rounds=max_rounds,
            seeds=seeds,
            hopeless_threshold=hopeless_threshold,
            trivial_threshold=trivial_threshold,
            spread_threshold=spread_threshold,
            env_kwargs=env_kwargs,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
        )

    cells: list[RegimeCellResult] = []
    grouped_best_anc: dict[tuple[float, float], list[float]] = {}
    grouped_cells: dict[tuple[float, float], list[tuple[int, dict[str, Any]]]] = {}

    for alpha in alpha_values:
        for pfail in pfail_values:
            for budget in budgets:
                ref_budget = budget if scale_budget else reference_budget
                policy_summaries = evaluate_policy_factories_with_optional_scaling(
                    graphs,
                    policy_factories,
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    max_rounds=max_rounds,
                    seeds=seeds,
                    env_kwargs=env_kwargs,
                    scale_budget=scale_budget,
                    scale_max_rounds=scale_max_rounds,
                    reference_budget=ref_budget,
                    reference_n=reference_n,
                )
                grouped_cells.setdefault((alpha, pfail), []).append((budget, policy_summaries))
                grouped_best_anc.setdefault((alpha, pfail), []).append(
                    max(summary.final_nc.mean for summary in policy_summaries.values())
                )

    for (alpha, pfail), budget_summaries in grouped_cells.items():
        anc_values = grouped_best_anc[(alpha, pfail)]
        budget_sensitivity = max(anc_values) - min(anc_values) if len(anc_values) > 1 else 0.0
        for budget, policy_summaries in budget_summaries:
            ref_budget = budget if scale_budget else reference_budget
            diagnostics = compute_regime_diagnostics(
                policy_summaries,
                hopeless_threshold=hopeless_threshold,
                trivial_threshold=trivial_threshold,
                spread_threshold=spread_threshold,
                budget_sensitivity=budget_sensitivity,
            )
            scaling_base = build_scaling_metadata(
                scale_budget=scale_budget,
                scale_max_rounds=scale_max_rounds,
                reference_budget=budget,
                reference_n=reference_n,
            )
            assert scaling_base is not None
            rep_nodes = graphs[0].number_of_nodes()
            res_b, res_mr = resolve_budget_and_rounds(
                num_nodes=rep_nodes,
                pfail=pfail,
                budget=budget,
                max_rounds=max_rounds,
                scale_budget=scale_budget,
                scale_max_rounds=scale_max_rounds,
                reference_budget=ref_budget,
                reference_n=reference_n,
            )
            cell_scaling = {
                **scaling_base,
                "representative_num_nodes": rep_nodes,
                "resolved_budget": res_b,
                "resolved_max_rounds": res_mr,
            }
            cells.append(
                RegimeCellResult(
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    diagnostics=diagnostics,
                    policy_summaries=dict(policy_summaries),
                    scaling=cell_scaling,
                )
            )

    return sorted(cells, key=lambda cell: (cell.alpha, cell.pfail, cell.budget))


def run_eval_set_mode(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from cascading_rl.evaluation.saved_eval_sets import (
        evaluate_policies_on_saved_instances,
        load_eval_instances,
        mean_final_nc_from_summaries,
    )

    log_path = getattr(args, "eval_set_log", None)
    log_file = None
    resolved_log: Path | None = None
    if log_path is not None:
        resolved_log = log_path if log_path.is_absolute() else ROOT / log_path
        resolved_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = resolved_log.open("w", encoding="utf-8")

    def p(*a: object, **kw: Any) -> None:
        print(*a, **kw)
        if log_file is not None:
            kw_log = dict(kw)
            kw_log["file"] = log_file
            kw_log["flush"] = True
            print(*a, **kw_log)

    try:
        eval_path = args.eval_set
        assert eval_path is not None
        if not eval_path.is_absolute():
            eval_path = ROOT / eval_path
        if not eval_path.exists():
            raise FileNotFoundError(f"Eval set not found: {eval_path}")

        instances = load_eval_instances(eval_path)

        scaled_eval_set = any(inst.get("b_scaled") is not None for inst in instances)
        if scaled_eval_set:
            bad_indices: list[int] = []
            for i, inst in enumerate(instances):
                raw = inst.get("b_scaled")
                if raw is None:
                    bad_indices.append(i)
                    continue
                try:
                    if int(raw) < 1:
                        bad_indices.append(i)
                except (TypeError, ValueError):
                    bad_indices.append(i)
            if bad_indices:
                raise ValueError(
                    f"{eval_path.name}: when any instance defines 'b_scaled', every instance must "
                    f"include a valid positive integer 'b_scaled'. "
                    f"Invalid or missing at indices {bad_indices[:10]!r}"
                    f"{'...' if len(bad_indices) > 10 else ''}."
                )

        ds_instances = [
            inst for inst in instances if inst.get("regime_label") == "decision-sensitive"
        ]
        if not ds_instances:
            warnings.warn(
                "Eval set contains no decision-sensitive instances — results are not meaningful "
                "for DS-focused analysis.",
                UserWarning,
                stacklevel=1,
            )

        training = config["training"]
        evaluation = config["evaluation"]
        env_kwargs = resolve_env_kwargs(config)
        selected = list(dict.fromkeys(args.policies or list(SUPPORTED_POLICIES)))
        if "rl" in selected and not args.checkpoint.exists():
            p(f"Warning: checkpoint missing at {args.checkpoint}; skipping rl policy.")
            selected = [policy for policy in selected if policy != "rl"]
        if not selected:
            raise ValueError("No policies left to evaluate on saved eval set.")
        factories = build_eval_policy_factories(
            args.checkpoint,
            base_seed=int(training["seed"]),
            selected_policies=selected,
        )
        overall, per_bucket = evaluate_policies_on_saved_instances(
            instances,
            factories,
            env_kwargs=env_kwargs,
            policy_names=selected,
        )

        p(f"=== Saved eval set: {eval_path}")
        p(f"instances={len(instances)}")
        per_budget = sorted(
            {
                int(inst["b_scaled"])
                if "b_scaled" in inst
                else int(inst["budget"])
                for inst in instances
            }
        )
        p(f"per-instance budgets used (unique, sorted): {per_budget}")
        for name in selected:
            if name not in overall:
                continue
            summary = overall[name]
            rws = summary.rounds_when_solved
            rws_s = f"{rws.mean:.2f}±{rws.stderr:.2f}" if rws is not None else "n/a"
            p(
                f"{name}: fully_restored={summary.fully_restored_count}/{summary.episode_count} "
                f"mean_rounds_if_restored={rws_s} "
                f"(mean_anc_all_episodes={summary.final_nc.mean:.3f}±{summary.final_nc.stderr:.3f})"
            )

        for label in sorted(per_bucket.keys()):
            n_label = sum(1 for inst in instances if str(inst.get("regime_label")) == label)
            p(f"\n[bucket: {label}] instances={n_label}")
            for name in selected:
                if name not in per_bucket[label]:
                    continue
                s = per_bucket[label][name]
                rws_b = s.rounds_when_solved
                rws_bs = f"{rws_b.mean:.2f}" if rws_b is not None else "n/a"
                p(
                    f"  {name}: restored={s.fully_restored_count}/{s.episode_count} "
                    f"mean_rounds_if_restored={rws_bs} final_nc_mean={s.final_nc.mean:.3f}"
                )

        large_names = {"large_graph_medium.pkl", "large_graph_large.pkl"}
        has_b_scaled = bool(instances) and any(
            inst.get("b_scaled") is not None for inst in instances
        )
        if eval_path.name in large_names or has_b_scaled:
            transfer_policies: list[str] = ["degree", "random"]
            if args.checkpoint.exists():
                transfer_policies = ["rl", "degree", "random"]
            t_factories = build_eval_policy_factories(
                args.checkpoint,
                base_seed=int(training["seed"]) + 1,
                selected_policies=transfer_policies,
            )
            t_names = list(t_factories.keys())

            baseline_path = ROOT / "eval_sets" / "ds_validation.pkl"
            table_rows: list[tuple[str, dict[str, float]]] = []
            if baseline_path.exists():
                base_instances = load_eval_instances(baseline_path)
                base_overall, _ = evaluate_policies_on_saved_instances(
                    base_instances,
                    t_factories,
                    env_kwargs=env_kwargs,
                    policy_names=t_names,
                )
                table_rows.append(
                    (
                        "validation (n~30-50)",
                        mean_final_nc_from_summaries(base_overall, t_names),
                    )
                )
            else:
                p(
                    "\nNote: eval_sets/ds_validation.pkl not found; transfer table has only "
                    "the current set."
                )

            cur_overall, _ = evaluate_policies_on_saved_instances(
                instances,
                t_factories,
                env_kwargs=env_kwargs,
                policy_names=t_names,
            )
            if eval_path.name == "large_graph_medium.pkl":
                row_label = "medium (n~100-150)"
            elif eval_path.name == "large_graph_large.pkl":
                row_label = "large (n~300-500)"
            else:
                row_label = f"current ({eval_path.name})"
            table_rows.append(
                (row_label, mean_final_nc_from_summaries(cur_overall, t_names)),
            )

            p("\n=== Zero-shot transfer (mean final_anc / PR)")
            header = "bucket\t" + "\t".join(t_names)
            p(header)
            for row_label, means in table_rows:
                vals = "\t".join(
                    f"{means.get(pol, float('nan')):.3f}" for pol in t_names
                )
                p(f"{row_label}\t{vals}")

        if resolved_log is not None:
            print(f"Saved eval-set report to {resolved_log}", flush=True)
    finally:
        if log_file is not None:
            log_file.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.eval_set is not None:
        run_eval_set_mode(args, config)
        return

    training = config["training"]
    regime = training["regime"]
    graph_cfg = training["graph"]
    evaluation = config["evaluation"]
    budget_scaling_cfg = config.get("budget_scaling", {})
    scale_budget_active = bool(args.scale_budget or budget_scaling_cfg.get("enabled", False))
    scale_max_rounds_cfg = budget_scaling_cfg.get("scale_max_rounds")
    scale_max_rounds_active = bool(
        args.scale_max_rounds
        if args.scale_max_rounds
        else (scale_max_rounds_cfg if scale_max_rounds_cfg is not None else True)
    )
    reference_n = int(
        args.reference_n
        if args.reference_n is not None
        else budget_scaling_cfg.get("reference_n", DEFAULT_REFERENCE_N)
    )
    if reference_n < 1:
        raise ValueError("--reference-n must be at least 1.")
    env_kwargs = resolve_env_kwargs(config)
    benchmark_reference_budget = int(regime["budget"])
    benchmark_graph_seed = (
        int(args.graph_seed) if args.graph_seed is not None else int(training["seed"]) + 1000
    )
    grid_spec = resolve_grid_spec(config, args)
    scaling_metadata = build_scaling_metadata(
        scale_budget=scale_budget_active,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=benchmark_reference_budget,
        reference_n=reference_n,
    )

    benchmark_policy_factories = build_eval_policy_factories(
        args.checkpoint,
        base_seed=benchmark_graph_seed,
        selected_policies=args.policies,
    )
    grid_policy_factories = build_eval_policy_factories(
        args.checkpoint,
        base_seed=int(grid_spec["graph_seed"]),
        selected_policies=args.policies,
    )
    rl_policy = (
        benchmark_policy_factories["rl"](0, 0)
        if "rl" in benchmark_policy_factories
        else None
    )
    selected_policy_names = list(benchmark_policy_factories.keys())

    graphs = make_graph_batch(
        num_graphs=int(training["benchmark_graphs"]),
        n_range=tuple(args.n_range) if args.n_range is not None else tuple(graph_cfg["n_range"]),
        m=int(graph_cfg["m"]),
        seed=benchmark_graph_seed,
    )

    target_solved_fraction = float(
        evaluation.get("minimum_budget_solved_target", evaluation.get("tau", 0.8))
    )
    log_scaling_decisions(
        graphs=graphs,
        pfail=float(regime["pfail"]),
        budget=benchmark_reference_budget,
        max_rounds=int(regime["max_rounds"]),
        scale_budget=scale_budget_active,
        scale_max_rounds=scale_max_rounds_active,
        reference_budget=benchmark_reference_budget,
        reference_n=reference_n,
    )
    summaries = evaluate_policy_factories_with_optional_scaling(
        graphs,
        benchmark_policy_factories,
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        budget=benchmark_reference_budget,
        max_rounds=int(regime["max_rounds"]),
        seeds=list(training["benchmark_seeds"]),
        env_kwargs=env_kwargs,
        scale_budget=scale_budget_active,
        scale_max_rounds=scale_max_rounds_active,
        reference_budget=benchmark_reference_budget,
        reference_n=reference_n,
    )

    representative_graph = graphs[0]
    evaluation_budgets = evaluation["budgets"]
    b_star = estimate_b_star_for_policies(
        policy_names=selected_policy_names,
        representative_graph=representative_graph,
        rl_policy=rl_policy,
        base_seed=int(training["seed"]),
        target_solved_fraction=target_solved_fraction,
        evaluation_budgets=list(evaluation_budgets),
        trials=len(training["benchmark_seeds"]),
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        max_rounds=int(regime["max_rounds"]),
        env_kwargs=env_kwargs,
        scale_budget=scale_budget_active,
        scale_max_rounds=scale_max_rounds_active,
        reference_n=reference_n,
    )
    serialized = {
        policy_name: serialize_policy_summary(summary, b_star=b_star.get(policy_name))
        for policy_name, summary in summaries.items()
    }
    serialized["scaling"] = scaling_metadata

    grid_budgets = [int(b) for b in grid_spec["budgets"]]
    grid_reference_budget = grid_budgets[0]
    grid_scaling_metadata = build_scaling_metadata(
        scale_budget=scale_budget_active,
        scale_max_rounds=scale_max_rounds_active,
        reference_budget=grid_reference_budget,
        reference_n=reference_n,
    )
    if (
        grid_scaling_metadata is not None
        and scale_budget_active
        and len(grid_budgets) > 1
    ):
        grid_scaling_metadata = {
            **grid_scaling_metadata,
            "reference_budget": None,
            "beta": None,
            "per_cell": True,
        }
    grid_graphs = make_graph_batch(
        num_graphs=grid_spec["num_graphs"],
        n_range=grid_spec["n_range"],
        m=grid_spec["m"],
        seed=grid_spec["graph_seed"],
    )
    threshold_cfg = config["regime_mapping"]
    log_scaling_decisions(
        graphs=grid_graphs,
        pfail=float(grid_spec["pfail_values"][0]),
        budget=int(grid_spec["budgets"][0]),
        max_rounds=grid_spec["max_rounds"],
        scale_budget=scale_budget_active,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=grid_reference_budget,
        reference_n=reference_n,
    )
    cells = build_regime_cells_with_optional_scaling(
        grid_graphs,
        grid_policy_factories,
        alpha_values=grid_spec["alpha_values"],
        pfail_values=grid_spec["pfail_values"],
        budgets=grid_spec["budgets"],
        max_rounds=grid_spec["max_rounds"],
        seeds=grid_spec["seeds"],
        hopeless_threshold=float(threshold_cfg["hopeless_threshold"]),
        trivial_threshold=float(threshold_cfg["trivial_threshold"]),
        spread_threshold=float(threshold_cfg["spread_threshold"]),
        env_kwargs=env_kwargs,
        scale_budget=scale_budget_active,
        scale_max_rounds=scale_max_rounds_active,
        reference_budget=grid_reference_budget,
        reference_n=reference_n,
    )
    grid_results = {
        "checkpoint": serialize_path(args.checkpoint),
        "grid_source": args.grid_source,
        "policies": selected_policy_names,
        "env": env_kwargs,
        "grid_spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in grid_spec.items()
        },
        "minimum_budget_solved_target": target_solved_fraction,
        "scaling": grid_scaling_metadata,
        "cells": [serialize_regime_cell(cell) for cell in cells],
        "bucket_summary": summarize_regime_buckets(cells),
    }

    output_dir = args.output_dir if args.output_dir is not None else ROOT / training["benchmark_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    grid_path = output_dir / "evaluation_grid_summary.json"
    regime_path = output_dir / "evaluation_regime_summary.json"
    metadata_path = output_dir / "run_metadata.json"
    summary_payload = {
        "checkpoint": serialize_path(args.checkpoint),
        "config": serialize_path(args.config),
        "policies": selected_policy_names,
        "minimum_budget_solved_target": target_solved_fraction,
        "env_stopping": env_kwargs,
        **serialized,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2)
    with grid_path.open("w", encoding="utf-8") as file:
        json.dump(grid_results, file, indent=2)
    with regime_path.open("w", encoding="utf-8") as file:
        json.dump(grid_results, file, indent=2)
    write_run_metadata(
        metadata_path,
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={
            "checkpoint": serialize_path(args.checkpoint),
            "grid_source": args.grid_source,
            "policies": selected_policy_names,
        },
    )

    print(f"Saved evaluation summary to {summary_path}")
    print(f"Saved grid evaluation summary to {grid_path}")
    print(f"Saved regime evaluation summary to {regime_path}")
    for policy_name, metrics in summary_payload.items():
        if policy_name in {
            "checkpoint",
            "config",
            "policies",
            "scaling",
            "minimum_budget_solved_target",
            "env_stopping",
        }:
            continue
        if not isinstance(metrics, dict) or "b_star" not in metrics:
            continue
        rws = metrics.get("rounds_when_solved")
        rws_s = f"{rws['mean']:.2f}" if isinstance(rws, dict) else "n/a"
        final_nc_mean = metrics["final_nc"]["mean"] if isinstance(metrics.get("final_nc"), dict) else metrics.get("final_anc_mean", 0.0)
        print(
            f"{policy_name}: fully_restored={metrics['fully_restored_count']}/"
            f"{metrics['episode_count']} mean_rounds_if_restored={rws_s} "
            f"(mean_nc_all={final_nc_mean:.3f}) b_star={metrics['b_star']}"
        )
    for bucket_name, bucket in grid_results["bucket_summary"].items():
        rl_gap = bucket["rl_vs_best_heuristic_gap"]
        gap_text = (
            f"{rl_gap['mean']:.3f}" if isinstance(rl_gap, dict) else "n/a"
        )
        print(
            f"[bucket:{bucket_name}] cells={bucket['cell_count']} "
            f"rl_minus_best_heuristic={gap_text}"
        )


if __name__ == "__main__":
    main()
