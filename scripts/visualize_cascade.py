from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Hashable

import matplotlib.pyplot as plt
import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import DEFAULT_REFERENCE_N, compute_scaled_budget
from cascading_rl.dynamics.cascade import (
    initialize_loads_and_capacities,
)
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.metrics.connectivity import accumulated_normalized_connectivity
from cascading_rl.policies import (
    choose_greedy_anc_node,
    choose_highest_betweenness_failed_node,
    choose_highest_degree_failed_node,
    choose_highest_overload_risk_node,
    choose_random_failed_node,
)


@dataclass(frozen=True)
class Frame:
    label: str
    observation: RecoveryObservation
    anc: float
    reward: float | None = None
    chosen_action: object | None = None
    highlighted_nodes: tuple[Hashable, ...] = ()


def build_policy(name: str, seed: int):
    rng = Random(seed)
    policy_map = {
        "random": lambda observation: choose_random_failed_node(observation, rng=rng),
        "degree": choose_highest_degree_failed_node,
        "risk": choose_highest_overload_risk_node,
        "greedy": choose_greedy_anc_node,
        "betweenness": choose_highest_betweenness_failed_node,
    }
    try:
        return policy_map[name]
    except KeyError as exc:
        raise ValueError(f"Unknown policy '{name}'.") from exc


def rollout_frames(
    env: RecoveryEnv,
    policy_name: str,
    seed: int,
) -> list[Frame]:
    policy = build_policy(policy_name, seed)
    loads, capacities = initialize_loads_and_capacities(env.base_graph, alpha=env.alpha)
    active = set(env.base_graph.nodes())

    frames = [
        Frame(
            label="Original graph",
            observation=RecoveryObservation(
                graph=env.base_graph,
                loads=dict(loads),
                capacities=dict(capacities),
                active=frozenset(active),
                failed=frozenset(),
                frontier=frozenset(),
                remaining_budget=env.budget,
                current_round=1,
            ),
            anc=accumulated_normalized_connectivity(env.base_graph, active),
        )
    ]

    observation = env.reset(seed=seed)
    frames.append(
        Frame(
            label="Initial random failures",
            observation=observation,
            anc=env.current_anc(),
            highlighted_nodes=tuple(sorted(observation.frontier)),
        )
    )

    while observation.failed:
        action = policy(observation)
        next_observation, reward, done, info = env.step(action)

        frames.append(
            Frame(
                label=(
                    f"Round {info['action_round']} repair {info['action_index_in_round']}"
                    if not info["cascade_executed"]
                    else f"Round {info['action_round']} cascade after repair {info['action_index_in_round']}"
                ),
                observation=next_observation,
                anc=float(info["anc"]),
                reward=reward,
                chosen_action=action,
                highlighted_nodes=(action,),
            )
        )

        observation = next_observation
        if done:
            break

    return frames


def draw_frame(ax, position: dict, frame: Frame) -> None:
    graph = frame.observation.graph
    active_nodes = set(frame.observation.active)
    failed_nodes = set(frame.observation.failed)

    node_colors = []
    for node in graph.nodes():
        if frame.chosen_action is not None and node == frame.chosen_action and node in active_nodes:
            node_colors.append("#457b9d")
        elif node in frame.highlighted_nodes and node in failed_nodes:
            node_colors.append("#ffb000")
        elif node in frame.observation.frontier and node in failed_nodes:
            node_colors.append("#ffb000")
        elif node in failed_nodes:
            node_colors.append("#d1495b")
        else:
            node_colors.append("#2a9d8f")

    nx.draw_networkx_edges(graph, position, ax=ax, edge_color="#aaaaaa", width=0.9, alpha=0.7)
    nx.draw_networkx_nodes(
        graph,
        position,
        ax=ax,
        node_color=node_colors,
        node_size=220,
        linewidths=0.5,
        edgecolors="#333333",
    )
    nx.draw_networkx_labels(graph, position, ax=ax, font_size=7)

    title_parts = [frame.label, f"ANC={frame.anc:.3f}"]
    if frame.chosen_action is None:
        title_parts.append(f"round={frame.observation.current_round}")
    if frame.observation.frontier:
        title_parts.append(f"frontier={len(frame.observation.frontier)}")
    if frame.reward is not None:
        title_parts.append(f"reward={frame.reward:.3f}")
    if frame.chosen_action is not None:
        title_parts.append(f"reactivated={frame.chosen_action}")
    ax.set_title("\n".join(title_parts), fontsize=10)
    ax.set_axis_off()


def plot_frames(frames: list[Frame], output_path: Path | None = None) -> None:
    graph = frames[0].observation.graph
    columns = min(3, len(frames))
    rows = math.ceil(len(frames) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows))
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    position = nx.spring_layout(graph, seed=42)

    for axis, frame in zip(axes_list, frames):
        draw_frame(axis, position, frame)
    for axis in axes_list[len(frames) :]:
        axis.set_axis_off()

    fig.suptitle("Cascade and recovery trajectory", fontsize=14)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a cascade and recovery trajectory.")
    parser.add_argument("--n", type=int, default=30, help="Number of nodes in the BA graph.")
    parser.add_argument("--m", type=int, default=2, help="Edges per new node in the BA graph.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Capacity tolerance parameter.")
    parser.add_argument("--pfail", type=float, default=0.1, help="Initial node failure probability.")
    parser.add_argument(
        "--budget",
        type=int,
        default=3,
        help="Reference recovery budget at the reference graph size.",
    )
    parser.add_argument(
        "--reference-n",
        type=int,
        default=DEFAULT_REFERENCE_N,
        help="Reference graph size used for canonical budget scaling.",
    )
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum number of repair rounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--policy",
        choices=["random", "degree", "risk", "greedy", "betweenness"],
        default="degree",
        help="Recovery heuristic used for the demo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, the figure is shown interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = make_ba_graph(n=args.n, m=args.m, seed=args.seed)
    resolved_budget = compute_scaled_budget(
        args.budget,
        num_nodes=graph.number_of_nodes(),
        reference_n=args.reference_n,
        enabled=True,
    )
    env = RecoveryEnv(
        graph,
        alpha=args.alpha,
        pfail=args.pfail,
        budget=resolved_budget,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    frames = rollout_frames(env, policy_name=args.policy, seed=args.seed)
    plot_frames(frames, output_path=args.output)


if __name__ == "__main__":
    main()
