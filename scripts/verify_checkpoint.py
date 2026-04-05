from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation.benchmarks import rollout_policy
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models.q_network import (
    QNetworkConfig,
    RecoveryQNetwork,
    build_greedy_policy,
    load_q_network,
)
from cascading_rl.policies import choose_random_failed_node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a trained checkpoint is not behaving like a random model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pt checkpoint file to verify.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config (default: config/default.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        _cfg = yaml.safe_load(f)

    # --- Load trained model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        trained_model, ckpt = load_q_network(args.checkpoint)
        model_config = trained_model.config
    except (RuntimeError, ValueError):
        # Legacy checkpoint: active_node_features may reference removed feature names, or
        # the stored input_dim may not match the saved weights. Strip feature-name lists and
        # infer input_dim from the weight tensor so resolve_feature_names() handles it.
        import dataclasses
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        known_fields = {f.name for f in dataclasses.fields(QNetworkConfig)}
        filtered = {k: v for k, v in ckpt["model_config"].items() if k in known_fields}
        filtered.pop("active_node_features", None)
        filtered.pop("active_global_features", None)
        first_weight_key = "encoder.layers.0.self_linear.weight"
        if first_weight_key in ckpt["model_state"]:
            filtered["input_dim"] = ckpt["model_state"][first_weight_key].shape[1]
            print(f"  (legacy checkpoint: inferred input_dim={filtered['input_dim']} from saved weights)")
        model_config = QNetworkConfig(**filtered)
        trained_model = RecoveryQNetwork(model_config)
        trained_model.load_state_dict(ckpt["model_state"])
        trained_model.eval()
    print(f"Model config: {model_config}")

    # --- Instantiate fresh untrained model with same architecture ---
    untrained_model = RecoveryQNetwork(model_config)
    untrained_model.eval()

    # --- Build graph + environment + observation ---
    graphs = make_graph_batch(num_graphs=1, n_range=(30, 32), m=2, seed=0)
    graph = graphs[0]
    env = RecoveryEnv(graph, alpha=0.1, pfail=0.15, budget=3, seed=42)
    obs = env.reset()

    # --- Score comparison ---
    print("\n=== Score Comparison ===")
    _, trained_scores = trained_model.score_observation(obs)
    _, untrained_scores = untrained_model.score_observation(obs)

    print(f"Trained scores  : {trained_scores}")
    print(f"Untrained scores: {untrained_scores}")

    identical = torch.allclose(trained_scores, untrained_scores)
    print(f"Scores identical: {identical}")
    if identical:
        print("WARNING: scores are identical — checkpoint may not have loaded correctly")

    # --- Build three policies ---
    rl_policy = build_greedy_policy(trained_model, batch_actions=True)
    untrained_policy = build_greedy_policy(untrained_model, batch_actions=True)
    rng = Random(0)
    random_policy = lambda observation: choose_random_failed_node(observation, rng=rng)

    # --- Episode env from config (mirrors training conditions exactly) ---
    regime = _cfg["training"]["regime"]
    ep_alpha = regime["alpha"]
    ep_pfail = regime["pfail"]
    ep_budget = regime["budget"]
    ep_max_rounds = regime.get("max_rounds")
    ep_capacity_noise = regime.get("capacity_noise", 0.0)
    ep_failure_bias = regime.get("failure_bias", "uniform")
    ep_action_space = regime.get("action_space", "failed")
    ep_obs_hops = regime.get("obs_hops")
    ep_abandonment = regime.get("abandonment_anc_threshold")
    print(f"\n=== Episode Evaluation (10 episodes, seeds 0–9) ===")
    print(f"  alpha={ep_alpha}, pfail={ep_pfail}, budget={ep_budget}, "
          f"max_rounds={ep_max_rounds}, abandonment_anc_threshold={ep_abandonment}")
    ep_env = RecoveryEnv(
        graph,
        alpha=ep_alpha,
        pfail=ep_pfail,
        budget=ep_budget,
        max_rounds=ep_max_rounds,
        seed=42,
        capacity_noise=ep_capacity_noise,
        failure_bias=ep_failure_bias,
        action_space=ep_action_space,
        obs_hops=ep_obs_hops,
        abandonment_anc_threshold=ep_abandonment,
    )

    results: dict[str, float] = {}
    for name, policy in [
        ("RL", rl_policy),
        ("Random", random_policy),
        ("Untrained", untrained_policy),
    ]:
        ancs = []
        for seed in range(10):
            result = rollout_policy(ep_env, policy, seed=seed)
            ancs.append(result.final_anc)
        mean_anc = sum(ancs) / len(ancs)
        print(f"{name:12s} mean final ANC: {mean_anc:.4f}")
        results[name] = mean_anc

    # --- Final diagnostic ---
    print()
    if abs(results["RL"] - results["Untrained"]) < 0.01:
        print(
            "WARNING: RL and Untrained ANC are within 0.01 — checkpoint is likely not training correctly"
        )
    else:
        print("Checkpoint appears to be trained (RL ANC differs meaningfully from untrained).")


if __name__ == "__main__":
    main()
