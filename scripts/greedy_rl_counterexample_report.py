"""Find a recovery instance where greedy fully restores the graph but the RL policy does not.

Writes trajectories (CSV), round-level comparison, ANC curves, and network keyframe figures
under experiments/evaluation_global/ (or a user-specified directory).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Hashable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cascading_rl.budgeting import DEFAULT_REFERENCE_N, compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import rollout_policy
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.models import build_greedy_policy, load_q_network
from cascading_rl.metrics.connectivity import accumulated_normalized_connectivity
from cascading_rl.policies import choose_greedy_anc_node
from visualize_cascade import Frame, draw_frame, plot_frames

Node = Hashable
PolicyFn = Callable[[RecoveryObservation], Node | list[Node]]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def env_kwargs_from_regime(regime: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "capacity_noise",
        "failure_bias",
        "action_space",
        "obs_hops",
        "abandonment_anc_threshold",
    )
    return {k: regime[k] for k in keys if k in regime}


def make_env(
    graph: nx.Graph,
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
    extra_env_kwargs: dict[str, Any],
) -> RecoveryEnv:
    n = graph.number_of_nodes()
    b = compute_scaled_budget(
        budget, num_nodes=n, reference_n=reference_n, enabled=scale_budget
    )
    mr = compute_scaled_max_rounds(
        max_rounds, num_nodes=n, reference_n=reference_n, enabled=scale_max_rounds
    )
    return RecoveryEnv(
        graph,
        alpha=alpha,
        pfail=pfail,
        budget=b,
        max_rounds=mr,
        **extra_env_kwargs,
    )


def _format_chosen(action: object) -> str:
    if isinstance(action, (list, tuple)):
        return ";".join(str(x) for x in action)
    return str(action)


def rollout_traced(
    env: RecoveryEnv,
    policy: PolicyFn,
    seed: int,
    *,
    policy_name: str,
) -> tuple[list[dict[str, Any]], dict[int | str, RecoveryObservation], bool]:
    observation = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    round_snaps: dict[int | str, RecoveryObservation] = {0: observation}
    transition = 0
    recovered = False

    if not observation.failed:
        return rows, round_snaps, True

    while observation.failed:
        pre_anc = env.current_anc()
        action = policy(observation)
        if isinstance(action, (list, tuple)):
            batch = list(action)
            observation, reward, done, info = env.step_batch(batch)
            intra = "batch"
        else:
            observation, reward, done, info = env.step(action)
            intra = str(info.get("action_index_in_round", ""))

        action_round = int(info["action_round"])
        rows.append(
            {
                "policy": policy_name,
                "transition_index": transition,
                "action_round": action_round,
                "intra_round": intra,
                "chosen_nodes": _format_chosen(action),
                "anc_pre_step": round(pre_anc, 6),
                "anc_after_reactivation": round(float(info["anc_after_reactivation"]), 6),
                "anc_post_cascade": round(float(info["anc_after_cascade"]), 6),
                "n_failed_after": int(info["failed_nodes"]),
                "cascade_executed": bool(info["cascade_executed"]),
                "round_complete": bool(info.get("round_complete", False)),
                "reward": round(float(reward), 6),
            }
        )
        transition += 1

        if isinstance(action, (list, tuple)):
            round_snaps[action_round] = observation
        elif info.get("round_complete"):
            round_snaps[action_round] = observation

        if done:
            recovered = not bool(observation.failed)
            break

    round_snaps["final"] = observation
    return rows, round_snaps, recovered


def round_end_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per completed action_round with end-of-round ANC and ordered node picks."""
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_round[int(row["action_round"])].append(row)

    out: list[dict[str, Any]] = []
    for r in sorted(by_round):
        steps = by_round[r]
        nodes_ordered: list[str] = []
        for s in steps:
            part = s["chosen_nodes"]
            if ";" in part:
                nodes_ordered.extend(part.split(";"))
            else:
                nodes_ordered.append(part)
        last = steps[-1]
        out.append(
            {
                "action_round": r,
                "chosen_nodes_in_order": ";".join(nodes_ordered),
                "anc_end_round": last["anc_post_cascade"],
                "n_failed_end_round": last["n_failed_after"],
            }
        )
    return out


def plot_anc_trajectories(
    greedy_rows: list[dict[str, Any]],
    rl_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, rows, style in (
        ("greedy", greedy_rows, "-"),
        ("rl", rl_rows, "--"),
    ):
        xs = [r["transition_index"] for r in rows]
        ys = [r["anc_post_cascade"] for r in rows]
        if rows:
            ax.plot(xs, ys, style, linewidth=2, label=f"{name} (post-cascade / transition)")
            ax.scatter(xs, ys, s=12, alpha=0.7)
    ax.set_xlabel("Transition index (greedy: one per round; RL: one per repair sub-step)")
    ax.set_ylabel("ANC")
    ax.set_title("ANC along each policy trajectory (same graph and failure seed)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def accumulated_normalized_connectivity_from_obs(obs: RecoveryObservation) -> float:
    return float(accumulated_normalized_connectivity(obs.graph, set(obs.active)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "learner" / "recovery_q.pt",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--graph-seed", type=int, default=42, help="BA generator seed.")
    p.add_argument("--n", type=int, default=40, help="BA graph size (moderate for greedy search).")
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--pfail", type=float, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--reference-n", type=int, default=DEFAULT_REFERENCE_N)
    p.add_argument("--scale-budget", action="store_true", default=True)
    p.add_argument("--no-scale-budget", action="store_false", dest="scale_budget")
    p.add_argument("--scale-max-rounds", action="store_true", default=True)
    p.add_argument("--no-scale-max-rounds", action="store_false", dest="scale_max_rounds")
    p.add_argument("--max-search-seeds", type=int, default=2500)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to experiments/evaluation_global/run_<utc-timestamp>.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    regime = cfg.get("training", {}).get("regime", {})
    alpha = float(args.alpha if args.alpha is not None else regime.get("alpha", 0.15))
    pfail = float(args.pfail if args.pfail is not None else regime.get("pfail", 0.18))
    budget = int(args.budget if args.budget is not None else regime.get("budget", 2))
    max_rounds = int(args.max_rounds if args.max_rounds is not None else regime.get("max_rounds", 20))
    extra = env_kwargs_from_regime(regime)

    graph = make_ba_graph(n=args.n, m=args.m, seed=args.graph_seed)

    def build_env() -> RecoveryEnv:
        return make_env(
            graph,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            scale_budget=args.scale_budget,
            scale_max_rounds=args.scale_max_rounds,
            reference_n=args.reference_n,
            extra_env_kwargs=extra,
        )

    device = torch.device(args.device)
    model, _ckpt = load_q_network(args.checkpoint, map_location=device)
    model.eval()
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)

    def greedy_policy(obs: RecoveryObservation) -> list[Node]:
        return choose_greedy_anc_node(obs)

    found_seed: int | None = None
    for fs in range(args.max_search_seeds):
        g_env = build_env()
        r_env = build_env()
        g_out = rollout_policy(g_env, greedy_policy, seed=fs)
        r_out = rollout_policy(r_env, rl_policy, seed=fs)
        if g_out.remaining_failed_nodes == 0 and r_out.remaining_failed_nodes > 0:
            found_seed = fs
            break

    if found_seed is None:
        print(
            "No instance found where greedy restores all nodes but RL does not. "
            "Try smaller --n, more --max-search-seeds, or different --graph-seed.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    out_dir = args.output_dir
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / "experiments" / "evaluation_global" / f"greedy_win_rl_loss_{ts}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    g_env = build_env()
    r_env = build_env()
    greedy_trace, greedy_snaps, _g_ok = rollout_traced(
        g_env, greedy_policy, found_seed, policy_name="greedy"
    )
    rl_trace, rl_snaps, _r_ok = rollout_traced(
        r_env, rl_policy, found_seed, policy_name="rl"
    )

    combined_trace = greedy_trace + rl_trace
    trace_path = out_dir / "step_trace.csv"
    fieldnames = list(combined_trace[0].keys()) if combined_trace else []
    with trace_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if combined_trace:
            w.writeheader()
            for row in combined_trace:
                w.writerow(row)

    g_sum = round_end_summary(greedy_trace)
    r_sum = round_end_summary(rl_trace)
    g_by = {int(row["action_round"]): row for row in g_sum}
    r_by = {int(row["action_round"]): row for row in r_sum}
    cmp_path = out_dir / "round_comparison.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "action_round",
            "greedy_chosen",
            "greedy_anc_end",
            "greedy_n_failed_end",
            "rl_chosen_in_order",
            "rl_anc_end",
            "rl_n_failed_end",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ar in sorted(set(g_by) | set(r_by)):
            gr = g_by.get(ar, {})
            rr = r_by.get(ar, {})
            writer.writerow(
                {
                    "action_round": ar,
                    "greedy_chosen": gr.get("chosen_nodes_in_order", ""),
                    "greedy_anc_end": gr.get("anc_end_round", ""),
                    "greedy_n_failed_end": gr.get("n_failed_end_round", ""),
                    "rl_chosen_in_order": rr.get("chosen_nodes_in_order", ""),
                    "rl_anc_end": rr.get("anc_end_round", ""),
                    "rl_n_failed_end": rr.get("n_failed_end_round", ""),
                }
            )

    meta = {
        "failure_seed": found_seed,
        "graph_seed": args.graph_seed,
        "n": args.n,
        "m": args.m,
        "alpha": alpha,
        "pfail": pfail,
        "budget_ref": budget,
        "max_rounds_ref": max_rounds,
        "scale_budget": args.scale_budget,
        "scale_max_rounds": args.scale_max_rounds,
        "reference_n": args.reference_n,
        "resolved_budget": build_env().budget,
        "resolved_max_rounds": build_env().max_rounds,
        "checkpoint": str(args.checkpoint if args.checkpoint.is_absolute() else (ROOT / args.checkpoint).resolve()),
        "greedy_recovered": True,
        "rl_recovered": not bool(rl_snaps["final"].failed),
        "extra_env_kwargs": extra,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    plot_anc_trajectories(greedy_trace, rl_trace, out_dir / "anc_trajectories.png")

    # Per-policy keyframe strip (full sequence can be long; use subset via plot_frames)
    def keyframe_frames(
        snaps: dict[int | str, RecoveryObservation],
        label_prefix: str,
        max_round: int,
    ) -> list[Frame]:
        chosen = [0]
        if max_round > 0:
            chosen.extend(
                sorted(
                    {1, max_round // 3, 2 * max_round // 3, max_round}
                    & set(range(1, max_round + 1))
                )
            )
        frames_out: list[Frame] = []
        for k in sorted(set(chosen)):
            obs = snaps[k]
            frames_out.append(
                Frame(
                    label=f"{label_prefix} — round {k if k else 'initial'}",
                    observation=obs,
                    anc=accumulated_normalized_connectivity_from_obs(obs),
                )
            )
        term = snaps.get("final")
        if term is not None and term is not snaps.get(max_round):
            frames_out.append(
                Frame(
                    label=f"{label_prefix} — terminal",
                    observation=term,
                    anc=accumulated_normalized_connectivity_from_obs(term),
                )
            )
        return frames_out

    g_round_max = max((k for k in greedy_snaps if isinstance(k, int)), default=0)
    r_round_max = max((k for k in rl_snaps if isinstance(k, int)), default=0)
    plot_frames(
        keyframe_frames(greedy_snaps, "Greedy", g_round_max),
        out_dir / "keyframes_greedy.png",
    )
    plot_frames(
        keyframe_frames(rl_snaps, "RL", r_round_max),
        out_dir / "keyframes_rl.png",
    )

    _write_aligned = out_dir / "keyframes_greedy_vs_rl.png"
    position = nx.spring_layout(graph, seed=42)
    int_keys_g = [k for k in greedy_snaps if isinstance(k, int)]
    Rg = max(int_keys_g) if int_keys_g else 0
    cols = sorted({0, max(1, Rg // 3), max(1, 2 * Rg // 3), Rg})
    n_cols = len(cols)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.0 * n_cols, 8.0))
    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]

    for c, rid in enumerate(cols):
        title = f"round {rid}" if rid > 0 else "initial"
        go = greedy_snaps.get(rid)
        if go is not None:
            draw_frame(
                axes[0][c],
                position,
                Frame(
                    label=f"Greedy {title}",
                    observation=go,
                    anc=accumulated_normalized_connectivity_from_obs(go),
                ),
            )
        ro = rl_snaps.get(rid)
        if ro is None:
            ro = rl_snaps.get("final")
            st = f"{title} (RL: no end-of-round {rid}; terminal)"
        else:
            st = title
        if ro is not None:
            draw_frame(
                axes[1][c],
                position,
                Frame(label=f"RL {st}", observation=ro, anc=accumulated_normalized_connectivity_from_obs(ro)),
            )
    fig.suptitle("Greedy (top) vs RL (bottom) at matched round indices", fontsize=13)
    fig.tight_layout()
    fig.savefig(_write_aligned, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote report under {out_dir}")


if __name__ == "__main__":
    main()
