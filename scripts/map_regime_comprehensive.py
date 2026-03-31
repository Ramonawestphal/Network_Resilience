from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

try:
    import pyarrow  # noqa: F401

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import compute_scaled_budget
from cascading_rl.dynamics.cascade import CascadeState, advance_cascade_round
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.metrics.connectivity import accumulated_normalized_connectivity
from cascading_rl.policies.betweenness_policy import choose_highest_betweenness_failed_node
from cascading_rl.policies.degree_policy import choose_highest_degree_failed_node
from cascading_rl.policies.random_policy import choose_random_failed_node

ALPHA_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
PFAIL_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
BUDGET_VALUES = [1, 2, 3, 4, 5, 6]
N_GRAPHS = 100
N_SEEDS = 10
GRAPH_N_RANGE = (30, 50)
GRAPH_M = 2
MAX_ROUNDS = 5
REFERENCE_N = 40
MASTER_SEED = 2026

DELTA_H = 0.30
DELTA_T = 0.80
DELTA_S = 0.15
MIN_DS_FRAC = 0.50

OUTPUT_DIR = "experiments/regime_comprehensive"

SENS_DELTA_H = [0.20, 0.25, 0.30, 0.35]
SENS_DELTA_T = [0.70, 0.75, 0.80, 0.85]
SENS_DELTA_S = [0.05, 0.10, 0.15, 0.20, 0.25]
SENS_MIN_DS = [0.30, 0.40, 0.50, 0.60]

POLICY_NAMES = ("degree", "random", "betweenness")
POLICY_ROW_COLUMNS = [
    "graph_id",
    "graph_seed",
    "n",
    "mean_degree",
    "max_degree",
    "alpha",
    "pfail",
    "budget_ref",
    "scaled_budget",
    "seed_index",
    "env_seed",
    "policy",
    "n_failed_at_start",
    "pr_post_cascade",
    "feasibility_ratio",
    "final_pr",
    "n_active_final",
    "solved",
    "spread_vs_random",
    "instance_label",
]
PNG_FILENAMES = (
    "spread_distribution_by_alpha.png",
    "pr_degree_distribution_by_alpha.png",
    "cascade_amplification_heatmap.png",
    "pr_post_cascade_heatmap.png",
    "ds_fraction_heatmap.png",
    "interestingness_heatmap.png",
    "budget_comparison.png",
    "feasibility_heatmap.png",
    "threshold_sensitivity_heatmap.png",
    "graph_vs_seed_variance.png",
)


@dataclass(frozen=True)
class MappingConfig:
    """Container for all grid, threshold, and runtime constants used by the script."""

    alpha_values: tuple[float, ...]
    pfail_values: tuple[float, ...]
    budget_values: tuple[int, ...]
    n_graphs: int
    n_seeds: int
    graph_n_range: tuple[int, int]
    graph_m: int
    max_rounds: int
    reference_n: int
    master_seed: int
    delta_h: float
    delta_t: float
    delta_s: float
    min_ds_frac: float
    sens_delta_h: tuple[float, ...]
    sens_delta_t: tuple[float, ...]
    sens_delta_s: tuple[float, ...]
    sens_min_ds: tuple[float, ...]
    output_dir: str

    @property
    def total_cells(self) -> int:
        return len(self.alpha_values) * len(self.pfail_values) * len(self.budget_values)

    @property
    def total_policy_rows(self) -> int:
        return self.total_cells * self.n_graphs * self.n_seeds * len(POLICY_NAMES)

    @property
    def rows_per_cell(self) -> int:
        return self.n_graphs * self.n_seeds * len(POLICY_NAMES)

    @property
    def grid_dict(self) -> dict[str, Any]:
        return {
            "alpha_values": list(self.alpha_values),
            "pfail_values": list(self.pfail_values),
            "budget_values": list(self.budget_values),
            "graph_n_range": list(self.graph_n_range),
            "graph_m": self.graph_m,
            "max_rounds": self.max_rounds,
            "reference_n": self.reference_n,
            "n_graphs": self.n_graphs,
            "n_seeds": self.n_seeds,
        }


def default_config() -> MappingConfig:
    """Return the production configuration backed by the module constants."""

    return MappingConfig(
        alpha_values=tuple(ALPHA_VALUES),
        pfail_values=tuple(PFAIL_VALUES),
        budget_values=tuple(BUDGET_VALUES),
        n_graphs=N_GRAPHS,
        n_seeds=N_SEEDS,
        graph_n_range=GRAPH_N_RANGE,
        graph_m=GRAPH_M,
        max_rounds=MAX_ROUNDS,
        reference_n=REFERENCE_N,
        master_seed=MASTER_SEED,
        delta_h=DELTA_H,
        delta_t=DELTA_T,
        delta_s=DELTA_S,
        min_ds_frac=MIN_DS_FRAC,
        sens_delta_h=tuple(SENS_DELTA_H),
        sens_delta_t=tuple(SENS_DELTA_T),
        sens_delta_s=tuple(SENS_DELTA_S),
        sens_min_ds=tuple(SENS_MIN_DS),
        output_dir=OUTPUT_DIR,
    )


def parse_args() -> argparse.Namespace:
    """Parse the minimal CLI used for help text and output-directory override."""

    parser = argparse.ArgumentParser(
        description="Run the comprehensive empirical regime mapping analysis."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override. The regime grid remains hard-coded.",
    )
    return parser.parse_args()


def timestamp_utc() -> str:
    """Return an ISO 8601 UTC timestamp for artifact metadata."""

    return datetime.now(timezone.utc).isoformat()


def print_startup_summary(config: MappingConfig) -> None:
    """Print the global grid size and policy-row count before any evaluation starts."""

    print(f"Total cells: {config.total_cells}")
    print(f"Total instances: {config.total_policy_rows}")


def build_graph_bank(config: MappingConfig) -> tuple[list[Any], pd.DataFrame]:
    """Generate the fixed BA graph bank once and keep it stable across all regime cells.

    Graph sizes are sampled deterministically from `MASTER_SEED`, while each BA graph
    itself uses the requested `graph_seed_i = MASTER_SEED * 1000 + i`. The returned
    metadata table is used later for invariants, aggregation, and reporting.
    """

    size_rng = Random(config.master_seed)
    min_n, max_n = config.graph_n_range
    graphs: list[Any] = []
    rows: list[dict[str, Any]] = []

    for graph_id in range(config.n_graphs):
        graph_seed = config.master_seed * 1000 + graph_id
        n = size_rng.randint(min_n, max_n)
        graph = make_ba_graph(n=n, m=config.graph_m, seed=graph_seed)
        graph.graph["graph_id"] = graph_id
        degrees = [degree for _, degree in graph.degree()]
        rows.append(
            {
                "graph_id": graph_id,
                "graph_seed": graph_seed,
                "n": graph.number_of_nodes(),
                "mean_degree": float(sum(degrees) / len(degrees)),
                "max_degree": int(max(degrees)),
            }
        )
        graphs.append(graph)

    graph_frame = pd.DataFrame(rows).sort_values("graph_id").reset_index(drop=True)
    assert len(graphs) == config.n_graphs
    assert graph_frame["n"].between(min_n, max_n).all()
    print(
        "Graph size summary: "
        f"mean={graph_frame['n'].mean():.2f}, "
        f"range=({int(graph_frame['n'].min())}, {int(graph_frame['n'].max())})"
    )
    return graphs, graph_frame


def env_seed_for_instance(config: MappingConfig, graph_id: int, seed_index: int) -> int:
    """Return the deterministic per-instance environment seed required by the spec."""

    return config.master_seed * 100000 + graph_id * 1000 + seed_index


def checkpoint_parquet_path(output_dir: Path) -> Path:
    """Return the parquet checkpoint path."""

    return output_dir / "checkpoint.parquet"


def checkpoint_csv_path(output_dir: Path) -> Path:
    """Return the CSV fallback checkpoint path."""

    return output_dir / "checkpoint.csv"


def build_run_metadata(config: MappingConfig, total_rows: int, *, timestamp: str) -> dict[str, Any]:
    """Create the run metadata payload required by the specification."""

    return {
        "generated_by": "map_regime_comprehensive.py",
        "master_seed": config.master_seed,
        "timestamp": timestamp,
        "n_graphs": config.n_graphs,
        "n_seeds": config.n_seeds,
        "total_instances": total_rows,
        "grid": config.grid_dict,
        "thresholds": {
            "delta_h": config.delta_h,
            "delta_t": config.delta_t,
            "delta_s": config.delta_s,
            "min_ds_frac": config.min_ds_frac,
        },
    }


def empty_policy_frame() -> pd.DataFrame:
    """Return an empty policy-row frame with the canonical checkpoint columns."""

    return pd.DataFrame(columns=POLICY_ROW_COLUMNS)


def load_checkpoint(output_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load the checkpoint from parquet if available, otherwise from CSV fallback."""

    parquet_path = checkpoint_parquet_path(output_dir)
    csv_path = checkpoint_csv_path(output_dir)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), "parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path), "csv"
    return empty_policy_frame(), "parquet" if PARQUET_AVAILABLE else "csv"


def save_checkpoint(frame: pd.DataFrame, output_dir: Path, checkpoint_kind: str) -> None:
    """Persist the checkpoint in parquet or CSV format depending on availability."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_kind == "parquet" and PARQUET_AVAILABLE:
        frame.to_parquet(checkpoint_parquet_path(output_dir), index=False)
    else:
        frame.to_csv(checkpoint_csv_path(output_dir), index=False)


def completed_cells_from_checkpoint(
    frame: pd.DataFrame,
    config: MappingConfig,
) -> tuple[set[tuple[float, float, int]], pd.DataFrame]:
    """Return completed cells and strip any incomplete cell rows from the checkpoint.

    A cell is complete only when it contains exactly `N_GRAPHS * N_SEEDS * 3` policy rows.
    Incomplete cell rows are dropped so a restarted run recomputes the cell from scratch
    and does not accumulate duplicates.
    """

    if frame.empty:
        return set(), frame
    grouped = frame.groupby(["alpha", "pfail", "budget_ref"]).size()
    complete = {
        (float(alpha), float(pfail), int(budget_ref))
        for (alpha, pfail, budget_ref), size in grouped.items()
        if int(size) == config.rows_per_cell
    }
    incomplete = {
        (float(alpha), float(pfail), int(budget_ref))
        for (alpha, pfail, budget_ref), size in grouped.items()
        if int(size) != config.rows_per_cell
    }
    if not incomplete:
        return complete, frame
    cleaned = frame.copy()
    for alpha, pfail, budget_ref in incomplete:
        cleaned = cleaned[
            ~(
                (cleaned["alpha"] == alpha)
                & (cleaned["pfail"] == pfail)
                & (cleaned["budget_ref"] == budget_ref)
            )
        ]
    return complete, cleaned.reset_index(drop=True)


def preview_post_cascade_metrics(state: CascadeState) -> tuple[int, float]:
    """Simulate the preview cascade to full settlement on a cloned state.

    The real environment starts recovery immediately after the exogenous failure event.
    For this analysis, we additionally record the severity after the entire preview
    cascade settles on a cloned copy of the reset state so the exported diagnostics
    reflect the full cascade consequences without altering the actual episode rollout.
    """

    preview = state.copy()
    while preview.frontier and preview.failed:
        advance_cascade_round(preview)
    failed_count = len(preview.failed)
    post_cascade_pr = accumulated_normalized_connectivity(preview.graph, preview.active)
    return failed_count, float(post_cascade_pr)


def policy_action(observation: RecoveryObservation, policy_name: str, *, rng: Random | None) -> Any:
    """Select an action for one of the three supported evaluation policies."""

    if policy_name == "degree":
        return choose_highest_degree_failed_node(observation)
    if policy_name == "betweenness":
        return choose_highest_betweenness_failed_node(observation)
    if policy_name == "random":
        assert rng is not None
        return choose_random_failed_node(observation, rng=rng)
    raise ValueError(f"Unsupported policy: {policy_name}")


def run_policy_episode(
    graph: Any,
    *,
    alpha: float,
    pfail: float,
    scaled_budget: int,
    max_rounds: int,
    env_seed: int,
    policy_name: str,
) -> dict[str, Any]:
    """Run a full recovery episode for one policy on one fixed instance."""

    env = RecoveryEnv(
        graph,
        alpha=alpha,
        pfail=pfail,
        budget=scaled_budget,
        max_rounds=max_rounds,
        seed=env_seed,
    )
    observation = env.reset(seed=env_seed)
    failed_at_reset = frozenset(observation.failed)
    active_at_reset = frozenset(observation.active)
    action_rng = Random(env_seed) if policy_name == "random" else None
    while observation.failed:
        action = policy_action(observation, policy_name, rng=action_rng)
        observation, _reward, done, _info = env.step(action)
        if done:
            break
    return {
        "failed_at_reset": failed_at_reset,
        "active_at_reset": active_at_reset,
        "final_pr": float(env.current_anc()),
        "n_active_final": int(len(env.state.active)) if env.state is not None else 0,
        "solved": bool(env.state is not None and not env.state.failed),
    }


def classify_instance_label(pr_degree: float, pr_random: float, spread: float, config: MappingConfig) -> str:
    """Apply the requested hopeless/trivial/decision-sensitive/ambiguous label logic."""

    if pr_degree < config.delta_h:
        return "hopeless"
    if pr_random > config.delta_t:
        return "trivial"
    if spread >= config.delta_s:
        return "decision_sensitive"
    return "ambiguous"


def evaluate_instance_rows(
    graph: Any,
    graph_meta: dict[str, Any],
    *,
    alpha: float,
    pfail: float,
    budget_ref: int,
    seed_index: int,
    config: MappingConfig,
) -> list[dict[str, Any]]:
    """Evaluate all three policies on one matched instance and return policy rows.

    All policies share the same graph, `(alpha, pfail, budget_ref)`, and the exact same
    `env_seed`. The function verifies that every policy sees the same reset state, then
    records a shared instance label and shared diagnostic fields across the three rows.
    """

    graph_id = int(graph_meta["graph_id"])
    env_seed = env_seed_for_instance(config, graph_id, seed_index)
    scaled_budget = compute_scaled_budget(
        budget_ref,
        num_nodes=int(graph_meta["n"]),
        reference_n=config.reference_n,
        enabled=True,
    )
    baseline_env = RecoveryEnv(
        graph,
        alpha=alpha,
        pfail=pfail,
        budget=scaled_budget,
        max_rounds=config.max_rounds,
        seed=env_seed,
    )
    baseline_obs = baseline_env.reset(seed=env_seed)
    n_failed_at_start, pr_post_cascade = preview_post_cascade_metrics(baseline_env.state)
    feasibility_ratio = n_failed_at_start / float(scaled_budget * config.max_rounds)

    policy_results: dict[str, dict[str, Any]] = {}
    for policy_name in POLICY_NAMES:
        result = run_policy_episode(
            graph,
            alpha=alpha,
            pfail=pfail,
            scaled_budget=scaled_budget,
            max_rounds=config.max_rounds,
            env_seed=env_seed,
            policy_name=policy_name,
        )
        assert result["failed_at_reset"] == frozenset(baseline_obs.failed)
        assert result["active_at_reset"] == frozenset(baseline_obs.active)
        policy_results[policy_name] = result

    pr_degree = float(policy_results["degree"]["final_pr"])
    pr_random = float(policy_results["random"]["final_pr"])
    instance_label = classify_instance_label(pr_degree, pr_random, pr_degree - pr_random, config)

    rows: list[dict[str, Any]] = []
    for policy_name in POLICY_NAMES:
        spread_vs_random = float(policy_results[policy_name]["final_pr"]) - pr_random
        rows.append(
            {
                "graph_id": graph_id,
                "graph_seed": int(graph_meta["graph_seed"]),
                "n": int(graph_meta["n"]),
                "mean_degree": float(graph_meta["mean_degree"]),
                "max_degree": int(graph_meta["max_degree"]),
                "alpha": float(alpha),
                "pfail": float(pfail),
                "budget_ref": int(budget_ref),
                "scaled_budget": int(scaled_budget),
                "seed_index": int(seed_index),
                "env_seed": int(env_seed),
                "policy": policy_name,
                "n_failed_at_start": int(n_failed_at_start),
                "pr_post_cascade": float(pr_post_cascade),
                "feasibility_ratio": float(feasibility_ratio),
                "final_pr": float(policy_results[policy_name]["final_pr"]),
                "n_active_final": int(policy_results[policy_name]["n_active_final"]),
                "solved": bool(policy_results[policy_name]["solved"]),
                "spread_vs_random": float(spread_vs_random),
                "instance_label": instance_label,
            }
        )
    return rows


def policy_rows_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a policy-row DataFrame in the canonical column order."""

    return pd.DataFrame(rows, columns=POLICY_ROW_COLUMNS)


def run_mapping_loop(
    config: MappingConfig,
    output_dir: Path,
    *,
    fail_after_cells: int | None = None,
) -> pd.DataFrame:
    """Run the full cell-by-cell evaluation loop with checkpoint resume support."""

    print_startup_summary(config)
    graphs, graph_frame = build_graph_bank(config)
    graph_meta_lookup = {
        int(row.graph_id): row._asdict() for row in graph_frame.itertuples(index=False)
    }

    checkpoint_frame, checkpoint_kind = load_checkpoint(output_dir)
    complete_cells, cleaned_checkpoint = completed_cells_from_checkpoint(checkpoint_frame, config)
    if len(cleaned_checkpoint) != len(checkpoint_frame):
        checkpoint_frame = cleaned_checkpoint
        save_checkpoint(checkpoint_frame, output_dir, checkpoint_kind)
    else:
        checkpoint_frame = cleaned_checkpoint

    if complete_cells:
        print(f"Resuming: {len(complete_cells)} of {config.total_cells} cells already complete")
    else:
        print("Starting fresh")

    all_cells = list(product(config.alpha_values, config.pfail_values, config.budget_values))
    completed_count = len(complete_cells)
    with tqdm(total=config.total_cells, initial=completed_count, desc="Cells", position=0) as outer:
        for alpha, pfail, budget_ref in all_cells:
            cell_key = (float(alpha), float(pfail), int(budget_ref))
            if cell_key in complete_cells:
                continue
            cell_rows: list[dict[str, Any]] = []
            desc = f"[alpha={alpha:.2f} pfail={pfail:.2f} B={budget_ref}]"
            with tqdm(
                total=config.n_graphs * config.n_seeds,
                desc=desc,
                position=1,
                leave=False,
                bar_format="{desc} |{bar}| ETA: {remaining}",
            ) as inner:
                for graph_id, graph in enumerate(graphs):
                    graph_meta = graph_meta_lookup[graph_id]
                    for seed_index in range(config.n_seeds):
                        cell_rows.extend(
                            evaluate_instance_rows(
                                graph,
                                graph_meta,
                                alpha=alpha,
                                pfail=pfail,
                                budget_ref=budget_ref,
                                seed_index=seed_index,
                                config=config,
                            )
                        )
                        inner.update(1)
            cell_frame = policy_rows_dataframe(cell_rows)
            if checkpoint_frame.empty:
                checkpoint_frame = cell_frame.copy()
            else:
                checkpoint_frame = pd.concat([checkpoint_frame, cell_frame], ignore_index=True)
            checkpoint_frame = checkpoint_frame.sort_values(
                by=["alpha", "pfail", "budget_ref", "graph_id", "seed_index", "policy"]
            ).reset_index(drop=True)
            save_checkpoint(checkpoint_frame, output_dir, checkpoint_kind)
            print(
                f"Saved cell (alpha={alpha:.2f}, pfail={pfail:.2f}, budget={budget_ref}): "
                f"checkpoint now has {len(checkpoint_frame)} rows"
            )
            outer.update(1)
            completed_count += 1
            if fail_after_cells is not None and completed_count >= fail_after_cells:
                raise RuntimeError("Intentional interruption after checkpoint save.")
    return checkpoint_frame


def build_instance_summary(policy_rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot policy rows into one row per `(graph, alpha, pfail, budget, seed)` instance."""

    index_columns = [
        "graph_id",
        "graph_seed",
        "n",
        "mean_degree",
        "max_degree",
        "alpha",
        "pfail",
        "budget_ref",
        "scaled_budget",
        "seed_index",
        "env_seed",
        "n_failed_at_start",
        "pr_post_cascade",
        "feasibility_ratio",
        "instance_label",
    ]
    pivot = policy_rows.pivot_table(
        index=index_columns,
        columns="policy",
        values=["final_pr", "n_active_final", "solved", "spread_vs_random"],
        aggfunc="first",
    )
    pivot.columns = [f"{left}_{right}" for left, right in pivot.columns]
    instance_summary = pivot.reset_index().sort_values(
        by=["alpha", "pfail", "budget_ref", "graph_id", "seed_index"]
    )
    instance_summary["spread_degree_random"] = (
        instance_summary["final_pr_degree"] - instance_summary["final_pr_random"]
    )
    instance_summary["spread_betweenness_random"] = (
        instance_summary["final_pr_betweenness"] - instance_summary["final_pr_random"]
    )
    return instance_summary.reset_index(drop=True)


def quantile_metrics(series: pd.Series, prefix: str) -> dict[str, float]:
    """Compute the requested mean/std/quantile summary for a numeric series."""

    return {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
        f"{prefix}_p10": float(series.quantile(0.10)),
        f"{prefix}_p25": float(series.quantile(0.25)),
        f"{prefix}_p50": float(series.quantile(0.50)),
        f"{prefix}_p75": float(series.quantile(0.75)),
        f"{prefix}_p90": float(series.quantile(0.90)),
    }


def aggregate_single_cell(
    cell_policy_rows: pd.DataFrame,
    cell_instances: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate one regime cell into flat summary stats and graph-variance details."""

    record: dict[str, Any] = {
        "alpha": float(cell_instances["alpha"].iloc[0]),
        "pfail": float(cell_instances["pfail"].iloc[0]),
        "budget_ref": int(cell_instances["budget_ref"].iloc[0]),
    }
    record.update(quantile_metrics(cell_instances["n_failed_at_start"], "n_failed"))
    record.update(quantile_metrics(cell_instances["pr_post_cascade"], "pr_post_cascade"))
    record["feasibility_ratio_mean"] = float(cell_instances["feasibility_ratio"].mean())
    record["feasibility_ratio_std"] = float(cell_instances["feasibility_ratio"].std(ddof=0))
    record["frac_infeasible"] = float((cell_instances["feasibility_ratio"] > 1.0).mean())

    label_fractions = cell_instances["instance_label"].value_counts(normalize=True)
    record["f_hopeless"] = float(label_fractions.get("hopeless", 0.0))
    record["f_trivial"] = float(label_fractions.get("trivial", 0.0))
    record["f_ds"] = float(label_fractions.get("decision_sensitive", 0.0))
    record["f_ambiguous"] = float(label_fractions.get("ambiguous", 0.0))

    for policy_name in POLICY_NAMES:
        prefix = f"final_pr_{policy_name}"
        policy_slice = cell_policy_rows.loc[cell_policy_rows["policy"] == policy_name, "final_pr"]
        record.update(quantile_metrics(policy_slice, prefix))
        record[f"solved_frac_{policy_name}"] = float(
            cell_policy_rows.loc[cell_policy_rows["policy"] == policy_name, "solved"].mean()
        )

    record["spread_degree_random_mean"] = float(cell_instances["spread_degree_random"].mean())
    record["spread_degree_random_std"] = float(cell_instances["spread_degree_random"].std(ddof=0))
    record["spread_degree_random_p25"] = float(cell_instances["spread_degree_random"].quantile(0.25))
    record["spread_degree_random_p75"] = float(cell_instances["spread_degree_random"].quantile(0.75))
    record["spread_betweenness_random_mean"] = float(
        cell_instances["spread_betweenness_random"].mean()
    )

    if record["f_hopeless"] > 0.50:
        record["cell_label"] = "hopeless"
    elif record["f_trivial"] > 0.50:
        record["cell_label"] = "trivial"
    elif record["f_ds"] >= MIN_DS_FRAC:
        record["cell_label"] = "decision_sensitive"
    else:
        record["cell_label"] = "mixed"

    ds_instances = cell_instances.loc[cell_instances["instance_label"] == "decision_sensitive"]
    record["interestingness_degree"] = float(
        record["f_ds"] * ds_instances["spread_degree_random"].mean()
    ) if not ds_instances.empty else 0.0
    record["interestingness_betweenness"] = float(
        record["f_ds"] * ds_instances["spread_betweenness_random"].mean()
    ) if not ds_instances.empty else 0.0

    degree_subset = cell_policy_rows.loc[cell_policy_rows["policy"] == "degree"]
    per_graph = (
        degree_subset.groupby("graph_id")
        .agg(
            graph_mean_pr_degree=("final_pr", "mean"),
            graph_std_pr_degree=("final_pr", lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0),
            graph_mean_nfailed=("n_failed_at_start", "mean"),
            graph_std_nfailed=(
                "n_failed_at_start",
                lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0,
            ),
        )
        .reset_index()
        .sort_values("graph_id")
    )
    variance_payload = {
        "alpha": record["alpha"],
        "pfail": record["pfail"],
        "budget_ref": record["budget_ref"],
        "across_graph_std_pr_degree": float(per_graph["graph_mean_pr_degree"].std(ddof=0)),
        "within_graph_std_pr_degree": float(per_graph["graph_std_pr_degree"].mean()),
        "across_graph_std_nfailed": float(per_graph["graph_mean_nfailed"].std(ddof=0)),
        "within_graph_std_nfailed": float(per_graph["graph_std_nfailed"].mean()),
        "per_graph_stats": per_graph.to_dict(orient="records"),
    }
    record["across_graph_std_pr_degree"] = variance_payload["across_graph_std_pr_degree"]
    record["within_graph_std_pr_degree"] = variance_payload["within_graph_std_pr_degree"]
    record["across_graph_std_nfailed"] = variance_payload["across_graph_std_nfailed"]
    record["within_graph_std_nfailed"] = variance_payload["within_graph_std_nfailed"]
    return record, variance_payload


def aggregate_regime_cells(
    policy_rows: pd.DataFrame,
    instance_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Aggregate policy rows and instance summaries into per-cell outputs."""

    cell_records: list[dict[str, Any]] = []
    variance_records: list[dict[str, Any]] = []
    cell_keys = ["alpha", "pfail", "budget_ref"]
    grouped_instances = instance_summary.groupby(cell_keys, sort=True)
    grouped_policy_rows = policy_rows.groupby(cell_keys, sort=True)
    for key, cell_instances in grouped_instances:
        cell_policy_rows = grouped_policy_rows.get_group(key)
        cell_record, variance_record = aggregate_single_cell(cell_policy_rows, cell_instances)
        cell_records.append(cell_record)
        variance_records.append(variance_record)
    cell_frame = pd.DataFrame(cell_records).sort_values(
        by=["budget_ref", "alpha", "pfail"]
    ).reset_index(drop=True)
    return cell_frame, variance_records


def aggregate_budget_summary(cell_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell outputs into one summary row per reference budget."""

    rows: list[dict[str, Any]] = []
    for budget_ref, group in cell_frame.groupby("budget_ref", sort=True):
        rows.append(
            {
                "budget_ref": int(budget_ref),
                "n_cells_ds": int((group["cell_label"] == "decision_sensitive").sum()),
                "n_cells_trivial": int((group["cell_label"] == "trivial").sum()),
                "n_cells_hopeless": int((group["cell_label"] == "hopeless").sum()),
                "n_cells_mixed": int((group["cell_label"] == "mixed").sum()),
                "mean_f_ds": float(group["f_ds"].mean()),
                "std_f_ds": float(group["f_ds"].std(ddof=0)) if len(group) > 1 else 0.0,
                "mean_interestingness": float(group["interestingness_degree"].mean()),
                "std_interestingness": float(group["interestingness_degree"].std(ddof=0))
                if len(group) > 1
                else 0.0,
                "mean_feasibility_ratio": float(group["feasibility_ratio_mean"].mean()),
                "frac_infeasible_cells": float((group["feasibility_ratio_mean"] > 1.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("budget_ref").reset_index(drop=True)


def relabel_instances(
    instance_summary: pd.DataFrame,
    *,
    delta_h: float,
    delta_t: float,
    delta_s: float,
) -> pd.Series:
    """Recompute instance labels for one threshold combination."""

    labels: list[str] = []
    for row in instance_summary.itertuples(index=False):
        labels.append(
            classify_instance_label(
                float(row.final_pr_degree),
                float(row.final_pr_random),
                float(row.spread_degree_random),
                default_config()
                if (
                    math.isclose(delta_h, DELTA_H)
                    and math.isclose(delta_t, DELTA_T)
                    and math.isclose(delta_s, DELTA_S)
                )
                else MappingConfig(
                    alpha_values=(),
                    pfail_values=(),
                    budget_values=(),
                    n_graphs=0,
                    n_seeds=0,
                    graph_n_range=(0, 0),
                    graph_m=0,
                    max_rounds=0,
                    reference_n=0,
                    master_seed=0,
                    delta_h=delta_h,
                    delta_t=delta_t,
                    delta_s=delta_s,
                    min_ds_frac=0.0,
                    sens_delta_h=(),
                    sens_delta_t=(),
                    sens_delta_s=(),
                    sens_min_ds=(),
                    output_dir="",
                ),
            )
        )
    return pd.Series(labels, index=instance_summary.index, dtype="object")


def sensitivity_label(
    pr_degree: float,
    pr_random: float,
    spread: float,
    *,
    delta_h: float,
    delta_t: float,
    delta_s: float,
) -> str:
    """Apply sensitivity-threshold labeling without building a full config object."""

    if pr_degree < delta_h:
        return "hopeless"
    if pr_random > delta_t:
        return "trivial"
    if spread >= delta_s:
        return "decision_sensitive"
    return "ambiguous"


def threshold_sensitivity_analysis(
    instance_summary: pd.DataFrame,
    config: MappingConfig,
) -> tuple[pd.DataFrame, tuple[float, float], dict[str, Any], dict[str, Any], list[list[int]]]:
    """Evaluate all threshold combinations and derive stability reports."""

    rows: list[dict[str, Any]] = []
    for delta_h, delta_t, delta_s, min_ds in product(
        config.sens_delta_h,
        config.sens_delta_t,
        config.sens_delta_s,
        config.sens_min_ds,
    ):
        relabeled = instance_summary.copy()
        relabeled["sens_label"] = relabeled.apply(
            lambda row: sensitivity_label(
                float(row["final_pr_degree"]),
                float(row["final_pr_random"]),
                float(row["spread_degree_random"]),
                delta_h=delta_h,
                delta_t=delta_t,
                delta_s=delta_s,
            ),
            axis=1,
        )
        cell_rows: list[dict[str, Any]] = []
        for (alpha, pfail, budget_ref), group in relabeled.groupby(
            ["alpha", "pfail", "budget_ref"], sort=True
        ):
            fractions = group["sens_label"].value_counts(normalize=True)
            f_ds = float(fractions.get("decision_sensitive", 0.0))
            if float(fractions.get("hopeless", 0.0)) > 0.50:
                cell_label = "hopeless"
            elif float(fractions.get("trivial", 0.0)) > 0.50:
                cell_label = "trivial"
            elif f_ds >= min_ds:
                cell_label = "decision_sensitive"
            else:
                cell_label = "mixed"
            ds_group = group.loc[group["sens_label"] == "decision_sensitive", "spread_degree_random"]
            cell_rows.append(
                {
                    "alpha": alpha,
                    "pfail": pfail,
                    "budget_ref": budget_ref,
                    "cell_label": cell_label,
                    "f_ds": f_ds,
                    "interestingness": float(f_ds * ds_group.mean()) if not ds_group.empty else 0.0,
                }
            )
        cell_frame = pd.DataFrame(cell_rows)
        counts = cell_frame["cell_label"].value_counts()
        rows.append(
            {
                "delta_h": float(delta_h),
                "delta_t": float(delta_t),
                "delta_s": float(delta_s),
                "min_ds_frac": float(min_ds),
                "n_cells_ds": int(counts.get("decision_sensitive", 0)),
                "n_cells_trivial": int(counts.get("trivial", 0)),
                "n_cells_hopeless": int(counts.get("hopeless", 0)),
                "n_cells_mixed": int(counts.get("mixed", 0)),
                "mean_f_ds_across_cells": float(cell_frame["f_ds"].mean()),
                "mean_interestingness_across_cells": float(cell_frame["interestingness"].mean()),
            }
        )

    sensitivity_frame = pd.DataFrame(rows).sort_values(
        by=["delta_h", "delta_t", "delta_s", "min_ds_frac"]
    ).reset_index(drop=True)
    proposed_subset = sensitivity_frame[
        (sensitivity_frame["delta_t"] == config.delta_t)
        & (sensitivity_frame["min_ds_frac"] == config.min_ds_frac)
    ]
    ds_by_delta_s = (
        proposed_subset[proposed_subset["delta_h"] == config.delta_h]
        .sort_values("delta_s")[["delta_s", "n_cells_ds"]]
        .reset_index(drop=True)
    )
    stable_values = [float(ds_by_delta_s.loc[0, "delta_s"])] if not ds_by_delta_s.empty else []
    for index in range(1, len(ds_by_delta_s)):
        if abs(int(ds_by_delta_s.loc[index, "n_cells_ds"]) - int(ds_by_delta_s.loc[index - 1, "n_cells_ds"])) < 5:
            stable_values.append(float(ds_by_delta_s.loc[index, "delta_s"]))
    stable_range = (min(stable_values), max(stable_values)) if stable_values else (math.nan, math.nan)

    most_permissive = (
        sensitivity_frame.sort_values(
            by=["n_cells_ds", "mean_f_ds_across_cells", "mean_interestingness_across_cells"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
    )
    proposed_entry = sensitivity_frame[
        (sensitivity_frame["delta_h"] == config.delta_h)
        & (sensitivity_frame["delta_t"] == config.delta_t)
        & (sensitivity_frame["delta_s"] == config.delta_s)
        & (sensitivity_frame["min_ds_frac"] == config.min_ds_frac)
    ].iloc[0].to_dict()

    matrix: list[list[int]] = []
    fixed_subset = sensitivity_frame[
        (sensitivity_frame["delta_t"] == config.delta_t)
        & (sensitivity_frame["min_ds_frac"] == config.min_ds_frac)
    ]
    for delta_h in config.sens_delta_h:
        row: list[int] = []
        for delta_s in config.sens_delta_s:
            match = fixed_subset[
                (fixed_subset["delta_h"] == delta_h) & (fixed_subset["delta_s"] == delta_s)
            ]
            row.append(int(match.iloc[0]["n_cells_ds"]))
        matrix.append(row)
    return sensitivity_frame, stable_range, most_permissive, proposed_entry, matrix


def flatten_cell_frame_for_csv(cell_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the already-flat cell frame in CSV-ready form."""

    return cell_frame.copy()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_instances_outputs(policy_rows: pd.DataFrame, output_dir: Path) -> None:
    """Save the full policy-row parquet and a truncated CSV sample."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if PARQUET_AVAILABLE:
        policy_rows.to_parquet(output_dir / "regime_instances.parquet", index=False)
    else:
        policy_rows.to_csv(output_dir / "regime_instances.parquet.csv", index=False)
    policy_rows.head(10000).to_csv(output_dir / "regime_instances.csv", index=False)


def save_figure(fig: Any, path: Path) -> None:
    """Save a figure at research-reporting quality and close it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def strip_jitter(values: list[float], position: float, rng: Random) -> list[float]:
    """Generate deterministic x-jitter values for strip overlays."""

    return [position + rng.uniform(-0.08, 0.08) for _ in values]


def plot_violin_by_alpha(
    instance_summary: pd.DataFrame,
    config: MappingConfig,
    *,
    value_column: str,
    title: str,
    filename: str,
    reference_lines: tuple[tuple[float, str, str], ...],
    ylabel: str,
) -> None:
    """Create the 3x3 alpha-by-budget violin plots used for spread and PR diagnostics."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    jitter_rng = Random(config.master_seed)
    for axis, alpha in zip(axes.flat, config.alpha_values, strict=False):
        alpha_frame = instance_summary[instance_summary["alpha"] == alpha]
        data = [
            alpha_frame.loc[alpha_frame["budget_ref"] == budget, value_column].tolist()
            for budget in config.budget_values
        ]
        axis.violinplot(data, positions=list(range(1, len(config.budget_values) + 1)), showmeans=True)
        for idx, values in enumerate(data, start=1):
            axis.scatter(
                strip_jitter(values, idx, jitter_rng),
                values,
                alpha=0.10,
                s=8,
                color="black",
            )
        for line_value, color, label in reference_lines:
            axis.axhline(line_value, color=color, linestyle="--", linewidth=1.2, label=label)
        pooled = alpha_frame[value_column]
        axis.axhline(float(pooled.quantile(0.25)), color="grey", linestyle=":", linewidth=1.0)
        axis.axhline(float(pooled.quantile(0.75)), color="grey", linestyle=":", linewidth=1.0)
        axis.set_title(f"α = {alpha:.2f}")
        axis.set_xticks(range(1, len(config.budget_values) + 1), [str(b) for b in config.budget_values])
        axis.set_xlabel("budget_ref")
        axis.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    save_figure(fig, Path(config.output_dir) / "plots" / filename)


def heatmap_matrix(
    frame: pd.DataFrame,
    *,
    row_values: tuple[float, ...],
    col_values: tuple[float, ...],
    value_column: str,
) -> list[list[float]]:
    """Build a rectangular matrix from alpha/pfail cell summaries."""

    matrix: list[list[float]] = []
    for pfail in row_values:
        row: list[float] = []
        for alpha in col_values:
            match = frame[(frame["alpha"] == alpha) & (frame["pfail"] == pfail)]
            row.append(float(match.iloc[0][value_column]) if not match.empty else math.nan)
        matrix.append(row)
    return matrix


def annotate_heatmap(axis: Any, matrix: list[list[float]]) -> None:
    """Write two-decimal annotations into a heatmap matrix."""

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            axis.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="black")


def plot_single_heatmap(
    matrix: list[list[float]],
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    filename: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a single annotated heatmap."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    annotate_heatmap(axis, matrix)
    axis.set_xticks(range(len(col_labels)), col_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("alpha")
    axis.set_ylabel("pfail")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    save_figure(fig, filename)


def plot_budget_panel_heatmaps(
    cell_frame: pd.DataFrame,
    config: MappingConfig,
    *,
    value_column: str,
    title: str,
    filename: str,
    cmap: str,
    highlight_ds: bool = False,
    star_top3: bool = False,
) -> None:
    """Save the 2x3 budget-panel heatmaps for DS fraction, interestingness, and infeasibility."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    vmax = float(cell_frame[value_column].max()) if value_column in cell_frame else None
    image = None
    for axis, budget_ref in zip(axes.flat, config.budget_values, strict=False):
        subset = cell_frame[cell_frame["budget_ref"] == budget_ref]
        matrix = heatmap_matrix(
            subset,
            row_values=config.pfail_values,
            col_values=config.alpha_values,
            value_column=value_column,
        )
        image = axis.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)
        annotate_heatmap(axis, matrix)
        axis.set_xticks(range(len(config.alpha_values)), [f"{v:.2f}" for v in config.alpha_values], rotation=45)
        axis.set_yticks(range(len(config.pfail_values)), [f"{v:.2f}" for v in config.pfail_values])
        axis.set_xlabel("alpha")
        axis.set_ylabel("pfail")
        axis.set_title(f"Budget ref = {budget_ref}")
        if highlight_ds:
            for row_idx, pfail in enumerate(config.pfail_values):
                for col_idx, alpha in enumerate(config.alpha_values):
                    match = subset[(subset["alpha"] == alpha) & (subset["pfail"] == pfail)]
                    if not match.empty and float(match.iloc[0]["f_ds"]) >= config.min_ds_frac:
                        axis.add_patch(Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, linewidth=2))
        if star_top3:
            top3 = subset.nlargest(3, value_column)[["alpha", "pfail"]]
            for _, row in top3.iterrows():
                col_idx = list(config.alpha_values).index(float(row["alpha"]))
                row_idx = list(config.pfail_values).index(float(row["pfail"]))
                axis.text(col_idx + 0.35, row_idx - 0.30, "*", fontsize=16, fontweight="bold")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    fig.suptitle(title)
    save_figure(fig, Path(config.output_dir) / "plots" / filename)


def plot_budget_comparison(budget_frame: pd.DataFrame, config: MappingConfig) -> None:
    """Render the grouped budget comparison bars with feasibility overlay."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(14, 8))
    x_values = list(range(len(budget_frame)))
    width = 0.22
    axis.bar([x - width for x in x_values], budget_frame["n_cells_ds"], width=width, label="n_cells_ds")
    axis.bar(x_values, budget_frame["mean_f_ds"] * 100.0, width=width, label="mean_f_ds (%)")
    axis.bar(
        [x + width for x in x_values],
        budget_frame["mean_interestingness"] * 100.0,
        width=width,
        label="mean_interestingness (%)",
    )
    axis.set_xticks(x_values, [str(int(value)) for value in budget_frame["budget_ref"]])
    axis.set_xlabel("budget_ref")
    axis.set_ylabel("Cell count / percentage")
    axis.set_title("Budget Comparison: Decision-Sensitivity vs Feasibility")
    axis.legend(loc="upper left")

    second_axis = axis.twinx()
    second_axis.plot(
        x_values,
        budget_frame["mean_feasibility_ratio"],
        color="red",
        marker="o",
        linewidth=2,
        label="mean_feasibility_ratio",
    )
    second_axis.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    second_axis.set_ylabel("Mean feasibility ratio")
    second_axis.legend(loc="upper right")
    save_figure(fig, Path(config.output_dir) / "plots" / "budget_comparison.png")


def plot_threshold_sensitivity_heatmap(
    matrix: list[list[int]],
    config: MappingConfig,
) -> None:
    """Create the `(delta_h, delta_s)` threshold sensitivity heatmap at fixed defaults."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(matrix, cmap="Purples", aspect="auto")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            axis.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")
    proposed_row = list(config.sens_delta_h).index(config.delta_h)
    proposed_col = list(config.sens_delta_s).index(config.delta_s)
    axis.add_patch(Rectangle((proposed_col - 0.5, proposed_row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
    axis.set_xticks(range(len(config.sens_delta_s)), [f"{v:.2f}" for v in config.sens_delta_s])
    axis.set_yticks(range(len(config.sens_delta_h)), [f"{v:.2f}" for v in config.sens_delta_h])
    axis.set_xlabel("delta_s")
    axis.set_ylabel("delta_h")
    axis.set_title("Threshold Sensitivity: n_cells_DS (fixed delta_t=0.80, min_ds=0.50)")
    fig.colorbar(image, ax=axis)
    save_figure(fig, Path(config.output_dir) / "plots" / "threshold_sensitivity_heatmap.png")


def plot_graph_vs_seed_variance(cell_frame: pd.DataFrame, config: MappingConfig) -> None:
    """Plot structural variance against stochastic variance for all regime cells."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(12, 8))
    scatter = axis.scatter(
        cell_frame["across_graph_std_pr_degree"],
        cell_frame["within_graph_std_pr_degree"],
        c=cell_frame["alpha"],
        s=40 + 180 * cell_frame["f_ds"],
        cmap="viridis",
        alpha=0.8,
    )
    upper = max(
        float(cell_frame["across_graph_std_pr_degree"].max()),
        float(cell_frame["within_graph_std_pr_degree"].max()),
    )
    axis.plot([0.0, upper], [0.0, upper], linestyle="--", color="black")
    top3 = cell_frame.nlargest(3, "across_graph_std_pr_degree")
    for row in top3.itertuples(index=False):
        axis.annotate(
            f"a={row.alpha:.2f}, p={row.pfail:.2f}, B={int(row.budget_ref)}",
            (row.across_graph_std_pr_degree, row.within_graph_std_pr_degree),
            textcoords="offset points",
            xytext=(6, 6),
        )
    axis.set_xlabel("across_graph_std_pr_degree")
    axis.set_ylabel("within_graph_std_pr_degree")
    axis.set_title("Structural vs Stochastic Variance in PR(degree)")
    fig.colorbar(scatter, ax=axis, label="alpha")
    save_figure(fig, Path(config.output_dir) / "plots" / "graph_vs_seed_variance.png")


def generate_plots(
    instance_summary: pd.DataFrame,
    cell_frame: pd.DataFrame,
    budget_frame: pd.DataFrame,
    sensitivity_matrix: list[list[int]],
    config: MappingConfig,
) -> None:
    """Generate all 10 required plots under `OUTPUT_DIR/plots/`."""

    output_dir = Path(config.output_dir)
    plot_violin_by_alpha(
        instance_summary,
        config,
        value_column="spread_degree_random",
        title="Spread Distribution by α (all pfail and budget pooled)",
        filename="spread_distribution_by_alpha.png",
        reference_lines=((config.delta_s, "red", "DELTA_S"),),
        ylabel="PR_degree - PR_random",
    )
    plot_violin_by_alpha(
        instance_summary,
        config,
        value_column="final_pr_degree",
        title="PR(degree) Distribution by α",
        filename="pr_degree_distribution_by_alpha.png",
        reference_lines=((config.delta_h, "red", "DELTA_H"), (config.delta_t, "blue", "DELTA_T")),
        ylabel="final_pr_degree",
    )
    pooled = instance_summary.groupby(["alpha", "pfail"], as_index=False).agg(
        n_failed_at_start=("n_failed_at_start", "median"),
        pr_post_cascade=("pr_post_cascade", "median"),
    )
    plot_single_heatmap(
        heatmap_matrix(
            pooled,
            row_values=config.pfail_values,
            col_values=config.alpha_values,
            value_column="n_failed_at_start",
        ),
        row_labels=[f"{v:.2f}" for v in config.pfail_values],
        col_labels=[f"{v:.2f}" for v in config.alpha_values],
        title="Median Failed Nodes at Recovery Start (post-cascade preview)",
        filename=output_dir / "plots" / "cascade_amplification_heatmap.png",
        cmap="YlOrRd",
    )
    plot_single_heatmap(
        heatmap_matrix(
            pooled,
            row_values=config.pfail_values,
            col_values=config.alpha_values,
            value_column="pr_post_cascade",
        ),
        row_labels=[f"{v:.2f}" for v in config.pfail_values],
        col_labels=[f"{v:.2f}" for v in config.alpha_values],
        title="Median PR Immediately After Cascade (preview)",
        filename=output_dir / "plots" / "pr_post_cascade_heatmap.png",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    plot_budget_panel_heatmaps(
        cell_frame,
        config,
        value_column="f_ds",
        title="Decision-Sensitive Fraction f_DS",
        filename="ds_fraction_heatmap.png",
        cmap="coolwarm",
        highlight_ds=True,
    )
    plot_budget_panel_heatmaps(
        cell_frame,
        config,
        value_column="interestingness_degree",
        title="Interestingness Score (f_DS × mean spread in DS instances)",
        filename="interestingness_heatmap.png",
        cmap="Blues",
        star_top3=True,
    )
    plot_budget_comparison(budget_frame, config)
    plot_budget_panel_heatmaps(
        cell_frame,
        config,
        value_column="frac_infeasible",
        title="Fraction of Structurally Infeasible Instances (ρ > 1)",
        filename="feasibility_heatmap.png",
        cmap="Reds",
    )
    plot_threshold_sensitivity_heatmap(sensitivity_matrix, config)
    plot_graph_vs_seed_variance(cell_frame, config)


def training_recommendation(
    cell_frame: pd.DataFrame,
    budget_frame: pd.DataFrame,
    stable_delta_s_range: tuple[float, float],
    proposed_threshold_entry: dict[str, Any],
    config: MappingConfig,
) -> dict[str, Any]:
    """Compute the best single cell, best budget, mixed regime, and stability verdict."""

    eligible = cell_frame[
        (cell_frame["cell_label"] == "decision_sensitive")
        & (cell_frame["feasibility_ratio_mean"] < 1.0)
        & (cell_frame["f_ds"] >= config.min_ds_frac)
    ]
    if eligible.empty:
        best_single = None
    else:
        top = eligible.sort_values("interestingness_degree", ascending=False).iloc[0]
        best_single = {
            "alpha": float(top["alpha"]),
            "pfail": float(top["pfail"]),
            "budget_ref": int(top["budget_ref"]),
            "f_ds": float(top["f_ds"]),
            "interestingness": float(top["interestingness_degree"]),
            "feasibility_ratio_mean": float(top["feasibility_ratio_mean"]),
        }

    best_budget_row = budget_frame.sort_values(
        by=["n_cells_ds", "mean_feasibility_ratio"],
        ascending=[False, True],
        key=lambda column: abs(column - 0.5) if column.name == "mean_feasibility_ratio" else column,
    ).iloc[0]
    best_budget_ref = int(best_budget_row["budget_ref"])

    budget_cells = cell_frame[cell_frame["budget_ref"] == best_budget_ref]
    ds_budget_cells = budget_cells[budget_cells["cell_label"] == "decision_sensitive"]
    if ds_budget_cells.empty:
        mixed_regime = {
            "budget_ref": best_budget_ref,
            "alpha_values": [],
            "pfail_values": [],
            "n_ds_cells_covered": 0,
        }
    else:
        threshold = 0.5 * float(ds_budget_cells["interestingness_degree"].max())
        selected = ds_budget_cells[ds_budget_cells["interestingness_degree"] >= threshold]
        mixed_regime = {
            "budget_ref": best_budget_ref,
            "alpha_values": sorted(selected["alpha"].drop_duplicates().tolist()),
            "pfail_values": sorted(selected["pfail"].drop_duplicates().tolist()),
            "n_ds_cells_covered": int(len(selected)),
        }

    stable_low, stable_high = stable_delta_s_range
    return {
        "best_single_cell": best_single,
        "best_budget_ref": best_budget_ref,
        "recommended_mixed_regime": mixed_regime,
        "threshold_stability": {
            "proposed_thresholds_in_stable_zone": (
                math.isfinite(stable_low) and stable_low <= config.delta_s <= stable_high
            ),
            "stable_delta_s_range": [stable_low, stable_high],
            "n_cells_ds_at_proposed": int(proposed_threshold_entry["n_cells_ds"]),
        },
    }


def print_budget_summary_table(budget_frame: pd.DataFrame) -> None:
    """Print the required per-budget summary table."""

    print("Budget | DS Cells | Trivial | Mixed | Mean f_DS | Mean Interest. | Mean Feasibility")
    print("-------|----------|---------|-------|-----------|----------------|------------------")
    for row in budget_frame.itertuples(index=False):
        print(
            f"{int(row.budget_ref):>6} | {int(row.n_cells_ds):>8} | {int(row.n_cells_trivial):>7} | "
            f"{int(row.n_cells_mixed):>5} | {float(row.mean_f_ds):>9.3f} | "
            f"{float(row.mean_interestingness):>14.3f} | {float(row.mean_feasibility_ratio):>16.3f}"
        )


def print_final_summary(
    policy_rows: pd.DataFrame,
    cell_frame: pd.DataFrame,
    recommendation: dict[str, Any],
    stable_delta_s_range: tuple[float, float],
) -> None:
    """Print the final research summary requested in the prompt."""

    counts = cell_frame["cell_label"].value_counts()
    print("=======================================================================")
    print("REGIME MAPPING COMPLETE")
    print("=======================================================================")
    print(f"Total instances evaluated: {len(policy_rows):,}")
    print(f"Total cells: {len(cell_frame)} (9 alpha x 7 pfail x 6 B)")
    print("Cell label distribution:")
    for label in ("decision_sensitive", "trivial", "hopeless", "mixed"):
        count = int(counts.get(label, 0))
        fraction = 100.0 * count / max(1, len(cell_frame))
        print(f"    {label.replace('_', '-').title():<19}: {count:>3}  ({fraction:>5.1f}%)")

    best_single = recommendation["best_single_cell"]
    if best_single is not None:
        print("\nBest single training cell:")
        print(
            f"    alpha={best_single['alpha']:.2f}, pfail={best_single['pfail']:.2f}, "
            f"budget_ref={best_single['budget_ref']}"
        )
        print(
            f"    f_DS={best_single['f_ds']:.3f}, "
            f"interestingness={best_single['interestingness']:.3f}, "
            f"feasibility={best_single['feasibility_ratio_mean']:.3f}"
        )

    mixed = recommendation["recommended_mixed_regime"]
    print(f"\nRecommended budget for mixed training: B_ref = {recommendation['best_budget_ref']}")
    print("Recommended (alpha, pfail) pairs for mixed training:")
    print(f"    alpha in {mixed['alpha_values']}, pfail in {mixed['pfail_values']}")

    stable = recommendation["threshold_stability"]["proposed_thresholds_in_stable_zone"]
    print("\nThreshold stability:")
    print(
        f"    Proposed thresholds (deltaH=0.30, deltaS=0.15) are "
        f"{'stable' if stable else 'unstable'}"
    )
    print(f"    Stable deltaS range: [{stable_delta_s_range[0]:.2f}, {stable_delta_s_range[1]:.2f}]")
    print("=======================================================================")


def assert_plot_outputs(output_dir: Path) -> None:
    """Confirm that all required plots exist and are materially non-empty."""

    plot_dir = output_dir / "plots"
    for filename in PNG_FILENAMES:
        path = plot_dir / filename
        assert path.exists(), f"Missing plot: {path}"
        assert path.stat().st_size > 10_000, f"Plot appears empty: {path}"


def run_analysis(
    config: MappingConfig | None = None,
    *,
    output_dir: Path | None = None,
    fail_after_cells: int | None = None,
) -> dict[str, Any]:
    """Run the full mapping pipeline and return all major intermediate artifacts."""

    config = config or default_config()
    resolved_output_dir = output_dir or (ROOT / config.output_dir)
    if output_dir is not None:
        config = MappingConfig(
            alpha_values=config.alpha_values,
            pfail_values=config.pfail_values,
            budget_values=config.budget_values,
            n_graphs=config.n_graphs,
            n_seeds=config.n_seeds,
            graph_n_range=config.graph_n_range,
            graph_m=config.graph_m,
            max_rounds=config.max_rounds,
            reference_n=config.reference_n,
            master_seed=config.master_seed,
            delta_h=config.delta_h,
            delta_t=config.delta_t,
            delta_s=config.delta_s,
            min_ds_frac=config.min_ds_frac,
            sens_delta_h=config.sens_delta_h,
            sens_delta_t=config.sens_delta_t,
            sens_delta_s=config.sens_delta_s,
            sens_min_ds=config.sens_min_ds,
            output_dir=str(resolved_output_dir),
        )

    policy_rows = run_mapping_loop(config, resolved_output_dir, fail_after_cells=fail_after_cells)
    expected_rows = config.total_policy_rows
    assert len(policy_rows) == expected_rows, f"Expected {expected_rows} policy rows, got {len(policy_rows)}"

    policy_rows = policy_rows.sort_values(
        by=["alpha", "pfail", "budget_ref", "graph_id", "seed_index", "policy"]
    ).reset_index(drop=True)
    save_instances_outputs(policy_rows, resolved_output_dir)

    run_metadata = build_run_metadata(config, len(policy_rows), timestamp=timestamp_utc())
    write_json(resolved_output_dir / "run_metadata.json", run_metadata)

    instance_summary = build_instance_summary(policy_rows)
    cell_frame, graph_variance = aggregate_regime_cells(policy_rows, instance_summary)
    assert len(cell_frame) == config.total_cells
    budget_frame = aggregate_budget_summary(cell_frame)
    sensitivity_frame, stable_delta_s_range, most_permissive, proposed_entry, sensitivity_matrix = threshold_sensitivity_analysis(
        instance_summary,
        config,
    )
    assert len(sensitivity_frame) == (
        len(config.sens_delta_h)
        * len(config.sens_delta_t)
        * len(config.sens_delta_s)
        * len(config.sens_min_ds)
    )

    write_json(
        resolved_output_dir / "regime_cells.json",
        {"metadata": run_metadata, "cells": cell_frame.to_dict(orient="records")},
    )
    flatten_cell_frame_for_csv(cell_frame).to_csv(
        resolved_output_dir / "regime_cells.csv", index=False
    )
    write_json(
        resolved_output_dir / "graph_variance.json",
        {"metadata": run_metadata, "cells": graph_variance},
    )
    write_json(
        resolved_output_dir / "budget_summary.json",
        {"metadata": run_metadata, "budgets": budget_frame.to_dict(orient="records")},
    )
    write_json(
        resolved_output_dir / "threshold_sensitivity.json",
        {
            "metadata": run_metadata,
            "sensitivity": sensitivity_frame.to_dict(orient="records"),
            "stable_delta_s_range": list(stable_delta_s_range),
            "most_permissive": most_permissive,
            "proposed_thresholds": proposed_entry,
        },
    )

    recommendation = training_recommendation(
        cell_frame,
        budget_frame,
        stable_delta_s_range,
        proposed_entry,
        config,
    )
    write_json(resolved_output_dir / "training_recommendation.json", recommendation)

    generate_plots(instance_summary, cell_frame, budget_frame, sensitivity_matrix, config)
    assert_plot_outputs(resolved_output_dir)

    print("\nTop 5 cells by interestingness_degree:")
    print(
        cell_frame.nlargest(5, "interestingness_degree")[
            ["alpha", "pfail", "budget_ref", "interestingness_degree", "f_ds", "cell_label"]
        ].to_string(index=False)
    )
    print("\nCell label distribution:")
    print(cell_frame["cell_label"].value_counts().to_string())
    print("\nBudget comparison summary:")
    print_budget_summary_table(budget_frame)
    print(
        f"\nBudget with most DS cells: {int(budget_frame.sort_values('n_cells_ds', ascending=False).iloc[0]['budget_ref'])}"
    )
    feasibility_rank = budget_frame.assign(
        feasibility_distance=(budget_frame["mean_feasibility_ratio"] - 0.5).abs()
    ).sort_values("feasibility_distance")
    print(
        "Budget with feasibility closest to 0.5: "
        f"{int(feasibility_rank.iloc[0]['budget_ref'])}"
    )
    print(
        f"\ndelta_S stable zone: [{stable_delta_s_range[0]:.2f}, {stable_delta_s_range[1]:.2f}] "
        "(n_cells_ds changes by < 5 within this range)"
    )
    print(
        f"Most permissive thresholds: dh={most_permissive['delta_h']:.2f}, "
        f"dt={most_permissive['delta_t']:.2f}, ds={most_permissive['delta_s']:.2f}, "
        f"min_ds={most_permissive['min_ds_frac']:.2f}"
    )
    print("\nn_cells_ds table for (delta_h, delta_s) at delta_t=0.80, min_ds=0.50:")
    header = "      " + " ".join(f"{delta_s:>6.2f}" for delta_s in config.sens_delta_s)
    print(header)
    for delta_h, row in zip(config.sens_delta_h, sensitivity_matrix, strict=False):
        print(f"{delta_h:>4.2f} " + " ".join(f"{value:>6d}" for value in row))
    print_final_summary(policy_rows, cell_frame, recommendation, stable_delta_s_range)

    return {
        "policy_rows": policy_rows,
        "instance_summary": instance_summary,
        "cell_frame": cell_frame,
        "budget_frame": budget_frame,
        "sensitivity_frame": sensitivity_frame,
        "recommendation": recommendation,
        "output_dir": resolved_output_dir,
    }


def main() -> None:
    """CLI entry point for the full comprehensive regime mapping run."""

    args = parse_args()
    run_analysis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
