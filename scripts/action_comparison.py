from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network
from cascading_rl.policies.degree_policy import choose_highest_degree_failed_node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RL and degree-policy actions on shared rollouts.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "curriculum_easy" / "recovery_q.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument("--num-graphs", type=int, default=10, help="Number of graphs to compare.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for graph generation and resets.")
    parser.add_argument("--alpha", type=float, default=0.20, help="Cascade alpha.")
    parser.add_argument("--pfail", type=float, default=0.10, help="Initial failure probability.")
    parser.add_argument("--budget", type=int, default=2, help="Recovery budget per round.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of rounds.")
    parser.add_argument(
        "--decision-steps",
        type=int,
        default=5,
        help="Number of first decisions to compare per graph.",
    )
    parser.add_argument(
        "--n-range",
        type=int,
        nargs=2,
        metavar=("MIN_N", "MAX_N"),
        default=(30, 50),
        help="Inclusive graph size range used by make_graph_batch.",
    )
    parser.add_argument("--m", type=int, default=2, help="Barabasi-Albert attachment parameter.")
    return parser.parse_args()


def reset_with_failures(env: RecoveryEnv, seed: int, rng: Random) -> RecoveryObservation:
    for _ in range(1024):
        observation = env.reset(seed=seed)
        if observation.failed:
            return observation
        seed = rng.randint(0, 10**9)
    raise RuntimeError("Could not sample an episode with at least one failed node.")


def main() -> None:
    args = parse_args()
    rng = Random(args.seed)
    graphs = make_graph_batch(
        num_graphs=args.num_graphs,
        n_range=tuple(args.n_range),
        m=args.m,
        seed=args.seed,
    )
    model, _ = load_q_network(args.checkpoint, map_location="cpu")
    rl_policy = build_greedy_policy(model, device="cpu")

    agreements: list[bool] = []
    compared_steps = 0
    sampled_graphs = 0

    for graph_index, graph in enumerate(graphs):
        env = RecoveryEnv(
            graph,
            alpha=args.alpha,
            pfail=args.pfail,
            budget=args.budget,
            max_rounds=args.max_rounds,
            seed=args.seed + graph_index,
        )
        observation = reset_with_failures(env, args.seed + graph_index, rng)
        sampled_graphs += 1
        print(f"graph={graph_index} failed={len(observation.failed)}")

        for step in range(args.decision_steps):
            if not observation.failed:
                break
            rl_action = rl_policy(observation)
            degree_action = choose_highest_degree_failed_node(observation)
            rl_degree = graph.degree(rl_action)
            degree_degree = graph.degree(degree_action)
            agrees = rl_action == degree_action
            agreements.append(agrees)
            compared_steps += 1
            print(
                f"  step={step + 1} "
                f"rl={rl_action} (degree={rl_degree})  "
                f"degree={degree_action} (degree={degree_degree})  "
                f"agree={agrees}"
            )
            observation, _, done, _ = env.step(rl_action)
            if done:
                break

    agreement_rate = sum(agreements) / len(agreements) if agreements else 0.0
    print("")
    print(f"graphs={sampled_graphs} compared_steps={compared_steps}")
    print(f"agreement_rate={agreement_rate:.1%}")


if __name__ == "__main__":
    main()
