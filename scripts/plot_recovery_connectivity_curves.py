"""Sample mean normalized connectivity (NC) per round for RL, all heuristics, and passive baseline.

Loads the trained policy from a checkpoint (default: config training checkpoint path) unless
``--no-rl`` is set. Training regime defaults come from config (alpha, pfail, budget, max_rounds,
BA graphs). Passive baseline: same initial state as recovery, then one cascade wave per round
with no reactivations (see ``cascading_rl.evaluation.passive_trajectory``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from random import Random
from statistics import mean, stdev

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation.benchmarks import EpisodeResult, collect_matched_episodes
from cascading_rl.evaluation.passive_trajectory import passive_nc_trajectory
from cascading_rl.evaluation.regime import build_policy_factories
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path


def load_checkpoint(path: Path) -> RecoveryQNetwork:
    import torch
    from cascading_rl.models import QNetworkConfig

    data = torch.load(path, map_location="cpu", weights_only=False)
    config = QNetworkConfig(**data["model_config"])
    model = RecoveryQNetwork(config)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model


def _aggregate_columns(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if not matrix:
        return [], []
    width = len(matrix[0])
    means: list[float] = []
    stderrs: list[float] = []
    for t in range(width):
        col = [float(row[t]) for row in matrix if t < len(row)]
        if not col:
            means.append(0.0)
            stderrs.append(0.0)
            continue
        m = mean(col)
        if len(col) > 1:
            stderrs.append(stdev(col) / sqrt(len(col)))
        else:
            stderrs.append(0.0)
        means.append(m)
    return means, stderrs


def _pad_heuristic_series(
    initial_nc: float,
    ep: EpisodeResult,
    horizon_rounds: int,
) -> list[float]:
    seq = [initial_nc] + [float(x) for x in ep.nc_by_round]
    solved = ep.remaining_failed_nodes == 0
    target = horizon_rounds + 1
    while len(seq) < target:
        seq.append(1.0 if solved else seq[-1])
    return seq[:target]


def _compute_initial_ncs(
    graphs: Sequence,
    seeds: Sequence[int],
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int | None,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
    env_kw: dict,
) -> list[float]:
    seeds_list = list(seeds)
    out: list[float] = []
    for graph in graphs:
        rb = compute_scaled_budget(
            budget,
            num_nodes=graph.number_of_nodes(),
            reference_n=reference_n,
            enabled=scale_budget,
        )
        rm = (
            compute_scaled_max_rounds(
                max_rounds,
                num_nodes=graph.number_of_nodes(),
                reference_n=reference_n,
                enabled=scale_max_rounds,
            )
            if max_rounds is not None
            else None
        )
        for seed in seeds_list:
            env = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=rb,
                max_rounds=rm,
                seed=seed,
                **env_kw,
            )
            env.reset(seed)
            out.append(env.current_nc())
    return out


def _horizon_rounds(
    graphs: Sequence,
    *,
    max_rounds: int | None,
    scale_max_rounds: bool,
    reference_n: int,
) -> int:
    if max_rounds is None:
        return max(g.number_of_nodes() for g in graphs)
    resolved = [
        compute_scaled_max_rounds(
            max_rounds,
            num_nodes=g.number_of_nodes(),
            reference_n=reference_n,
            enabled=scale_max_rounds,
        )
        for g in graphs
    ]
    return max(resolved)


def _passive_series_for_episode(
    graph,
    seed: int,
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int | None,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
    env_kw: dict,
    horizon_rounds: int,
) -> list[float]:
    rb = compute_scaled_budget(
        budget,
        num_nodes=graph.number_of_nodes(),
        reference_n=reference_n,
        enabled=scale_budget,
    )
    rm = (
        compute_scaled_max_rounds(
            max_rounds,
            num_nodes=graph.number_of_nodes(),
            reference_n=reference_n,
            enabled=scale_max_rounds,
        )
        if max_rounds is not None
        else None
    )
    env = RecoveryEnv(
        graph,
        alpha=alpha,
        pfail=pfail,
        budget=rb,
        max_rounds=rm,
        seed=seed,
        **env_kw,
    )
    env.reset(seed)
    if env.state is None:
        raise RuntimeError("env.state missing after reset")
    raw = passive_nc_trajectory(env.state, max_rounds=horizon_rounds)
    while len(raw) < horizon_rounds + 1:
        raw.append(raw[-1])
    return raw[: horizon_rounds + 1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    p.add_argument("--num-graphs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--graph-seed", type=int, default=1234, help="RNG seed for BA graph generation.")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--pfail", type=float, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--m", type=int, default=None, help="BA parameter m (default: training.graph.m).")
    p.add_argument("--n-min", type=int, default=None, help="Override training.graph.n_range lower bound.")
    p.add_argument("--n-max", type=int, default=None, help="Override training.graph.n_range upper bound.")
    p.add_argument("--reference-n", type=int, default=None, help="For budget scaling (default: config).")
    p.add_argument("--scale-budget", action="store_true", help="Scale budget with graph size.")
    p.add_argument("--scale-max-rounds", action="store_true", help="Scale max_rounds with graph size.")
    p.add_argument("--output", type=Path, default=ROOT / "experiments" / "recovery_curves" / "nc_by_round.png")
    p.add_argument("--json-out", type=Path, default=None, help="Optional JSON path for series + metadata.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained policy weights (default: training.checkpoint_dir / checkpoint_name from config).",
    )
    p.add_argument(
        "--no-rl",
        action="store_true",
        help="Skip loading RL; plot heuristics and passive only.",
    )
    p.add_argument("--no-stderr-bands", action="store_true", help="Do not draw stderr bands for policy curves.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training = cfg["training"]
    regime = training["regime"]
    graph_cfg = training["graph"]
    budget_scaling = cfg.get("budget_scaling", {})

    alpha = float(args.alpha if args.alpha is not None else regime["alpha"])
    pfail = float(args.pfail if args.pfail is not None else regime["pfail"])
    budget = int(args.budget if args.budget is not None else regime["budget"])
    max_rounds = int(args.max_rounds if args.max_rounds is not None else regime["max_rounds"])
    m = int(args.m if args.m is not None else graph_cfg["m"])
    n_lo = int(args.n_min if args.n_min is not None else graph_cfg["n_range"][0])
    n_hi = int(args.n_max if args.n_max is not None else graph_cfg["n_range"][1])
    reference_n = int(
        args.reference_n if args.reference_n is not None else budget_scaling.get("reference_n", 40)
    )
    scale_budget = bool(args.scale_budget or budget_scaling.get("enabled", False))
    scale_max_rounds = bool(args.scale_max_rounds or budget_scaling.get("scale_max_rounds", True))

    env_kw = {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": regime.get("obs_hops"),
        "abandonment_nc_threshold": regime.get("abandonment_nc_threshold"),
    }

    rng = Random(args.graph_seed)
    graphs = [
        make_ba_graph(n=rng.randint(n_lo, n_hi), m=m, seed=rng.randint(0, 10**9))
        for _ in range(args.num_graphs)
    ]

    horizon = _horizon_rounds(
        graphs,
        max_rounds=max_rounds,
        scale_max_rounds=scale_max_rounds,
        reference_n=reference_n,
    )

    checkpoint_path: Path | None = None
    if not args.no_rl:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = ROOT / training["checkpoint_dir"] / training["checkpoint_name"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RL checkpoint not found: {checkpoint_path}\n"
                "Train a policy, pass --checkpoint, or use --no-rl for heuristics only."
            )
        import torch

        device = torch.device("cpu")
        model = load_checkpoint(checkpoint_path)
        rl_policy = build_greedy_policy(model, device=device, batch_actions=False)
        factories = {
            "rl": lambda _gi, _se: rl_policy,
            **build_policy_factories(base_seed=0),
        }
    else:
        factories = build_policy_factories(base_seed=0)

    policy_names = list(factories.keys())

    episode_by_policy = collect_matched_episodes(
        graphs,
        factories,
        alpha=alpha,
        pfail=pfail,
        budget=budget,
        max_rounds=max_rounds,
        seeds=list(args.seeds),
        env_kwargs=env_kw,
        scale_budget=scale_budget,
        scale_max_rounds=scale_max_rounds,
        reference_n=reference_n,
    )

    initial_ncs = _compute_initial_ncs(
        graphs,
        args.seeds,
        alpha=alpha,
        pfail=pfail,
        budget=budget,
        max_rounds=max_rounds,
        scale_budget=scale_budget,
        scale_max_rounds=scale_max_rounds,
        reference_n=reference_n,
        env_kw=env_kw,
    )

    curves_mean: dict[str, list[float]] = {}
    curves_stderr: dict[str, list[float]] = {}

    for pname in policy_names:
        eps = episode_by_policy[pname]
        rows: list[list[float]] = []
        for i, ep in enumerate(eps):
            rows.append(_pad_heuristic_series(initial_ncs[i], ep, horizon))
        curves_mean[pname], curves_stderr[pname] = _aggregate_columns(rows)

    passive_rows: list[list[float]] = []
    for graph in graphs:
        for seed in args.seeds:
            passive_rows.append(
                _passive_series_for_episode(
                    graph,
                    seed,
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    max_rounds=max_rounds,
                    scale_budget=scale_budget,
                    scale_max_rounds=scale_max_rounds,
                    reference_n=reference_n,
                    env_kw=env_kw,
                    horizon_rounds=horizon,
                )
            )
    curves_mean["passive"], curves_stderr["passive"] = _aggregate_columns(passive_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    x = list(range(horizon + 1))
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=args.dpi)
    colors = plt.cm.tab10(range(10))

    plot_order = ["rl", "random", "degree", "greedy", "betweenness", "risk"]
    for idx, name in enumerate([n for n in plot_order if n in curves_mean]):
        ym = curves_mean[name]
        ye = curves_stderr[name]
        c = colors[idx % 10]
        lw = 2.4 if name == "rl" else 1.8
        ax.plot(x, ym, label=name, color=c, linewidth=lw)
        if not args.no_stderr_bands and len(ym) == len(ye):
            lo = [a - b for a, b in zip(ym, ye)]
            hi = [a + b for a, b in zip(ym, ye)]
            ax.fill_between(x, lo, hi, color=c, alpha=0.15)

    pm = curves_mean["passive"]
    ax.plot(x, pm, label="passive (no recovery)", color="black", linestyle="--", linewidth=2.0)

    ax.set_xlabel("Round (0 = after initial failures)")
    ax.set_ylabel("Normalized connectivity (NC)")
    ax.set_title(
        f"Mean NC vs round  |  α={alpha}, p_fail={pfail}, B={budget}, "
        f"max_rounds≤{horizon}  |  BA m={m}, n∈[{n_lo},{n_hi}]"
    )
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)

    meta = {
        "alpha": alpha,
        "pfail": pfail,
        "budget_ref": budget,
        "max_rounds_ref": max_rounds,
        "horizon_rounds": horizon,
        "scale_budget": scale_budget,
        "scale_max_rounds": scale_max_rounds,
        "reference_n": reference_n,
        "m": m,
        "n_range": [n_lo, n_hi],
        "num_graphs": args.num_graphs,
        "seeds": list(args.seeds),
        "graph_seed": args.graph_seed,
        "rl_included": not args.no_rl,
        "checkpoint_path": portable_artifact_path(checkpoint_path) if checkpoint_path is not None else None,
        "policies": list(curves_mean.keys()),
        "curves_mean": curves_mean,
        "curves_stderr": curves_stderr,
        "figure_path": portable_artifact_path(args.output),
    }
    json_path = args.json_out
    if json_path is None:
        json_path = args.output.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.output}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
