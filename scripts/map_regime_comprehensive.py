from __future__ import annotations

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
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.budgeting import compute_scaled_budget
from cascading_rl.dynamics.cascade import CascadeState, advance_cascade_round
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation import build_policy_factories
from cascading_rl.graph.generation import make_ba_graph

GENERATED_BY = "map_regime_comprehensive.py"

ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PFAIL_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
BUDGET_VALUES = [2, 3, 4, 5]
N_GRAPHS = 50
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

DELTA_H_CANDIDATES = [0.20, 0.25, 0.30, 0.35]
DELTA_T_CANDIDATES = [0.70, 0.75, 0.80, 0.85]
DELTA_S_CANDIDATES = [0.05, 0.10, 0.15, 0.20]
MIN_DS_FRAC_CANDIDATES = [0.30, 0.40, 0.50, 0.60]

OUTPUT_DIR = "experiments/regime_comprehensive"

INSTANCE_COLUMNS = [
    "graph_id",
    "graph_seed",
    "n",
    "m",
    "mean_degree",
    "max_degree",
    "alpha",
    "pfail",
    "budget_ref",
    "scaled_budget",
    "seed",
    "anc_degree",
    "anc_random",
    "spread",
    "n_initial_failures",
    "n_post_cascade_failures",
    "instance_label",
    "feasibility_ratio",
]


@dataclass(frozen=True)
class MappingConfig:
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
    delta_h_candidates: tuple[float, ...]
    delta_t_candidates: tuple[float, ...]
    delta_s_candidates: tuple[float, ...]
    min_ds_frac_candidates: tuple[float, ...]
    output_dir: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.n_seeds))

    @property
    def cell_count(self) -> int:
        return len(self.alpha_values) * len(self.pfail_values) * len(self.budget_values)

    def grid_dict(self) -> dict[str, Any]:
        return {
            "alpha_values": list(self.alpha_values),
            "pfail_values": list(self.pfail_values),
            "budget_values": list(self.budget_values),
            "n_graphs": self.n_graphs,
            "n_seeds": self.n_seeds,
            "graph_n_range": list(self.graph_n_range),
            "graph_m": self.graph_m,
            "max_rounds": self.max_rounds,
            "reference_n": self.reference_n,
            "delta_h": self.delta_h,
            "delta_t": self.delta_t,
            "delta_s": self.delta_s,
            "min_ds_frac": self.min_ds_frac,
            "delta_h_candidates": list(self.delta_h_candidates),
            "delta_t_candidates": list(self.delta_t_candidates),
            "delta_s_candidates": list(self.delta_s_candidates),
            "min_ds_frac_candidates": list(self.min_ds_frac_candidates),
            "output_dir": self.output_dir,
        }


def default_config() -> MappingConfig:
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
        delta_h_candidates=tuple(DELTA_H_CANDIDATES),
        delta_t_candidates=tuple(DELTA_T_CANDIDATES),
        delta_s_candidates=tuple(DELTA_S_CANDIDATES),
        min_ds_frac_candidates=tuple(MIN_DS_FRAC_CANDIDATES),
        output_dir=OUTPUT_DIR,
    )


def safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=0))


def build_metadata(config: MappingConfig, timestamp: str) -> dict[str, Any]:
    return {
        "generated_by": GENERATED_BY,
        "master_seed": config.master_seed,
        "timestamp": timestamp,
        "grid": config.grid_dict(),
    }


def parquet_metadata(metadata: dict[str, Any]) -> dict[bytes, bytes]:
    return {
        b"generated_by": str(metadata["generated_by"]).encode("utf-8"),
        b"master_seed": str(metadata["master_seed"]).encode("utf-8"),
        b"timestamp": str(metadata["timestamp"]).encode("utf-8"),
        b"grid": json.dumps(metadata["grid"], sort_keys=True).encode("utf-8"),
    }


def png_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "generated_by": str(metadata["generated_by"]),
        "master_seed": str(metadata["master_seed"]),
        "timestamp": str(metadata["timestamp"]),
        "grid": json.dumps(metadata["grid"], sort_keys=True),
    }


def default_output_dir(config: MappingConfig) -> Path:
    return ROOT / config.output_dir


def build_graph_bank(config: MappingConfig) -> tuple[list[Any], pd.DataFrame]:
    rng = Random(config.master_seed)
    graphs: list[Any] = []
    graph_rows: list[dict[str, Any]] = []
    min_n, max_n = config.graph_n_range

    for graph_id in range(config.n_graphs):
        graph_seed = config.master_seed * 1000 + graph_id
        n = rng.randint(min_n, max_n)
        graph = make_ba_graph(n=n, m=config.graph_m, seed=graph_seed)
        graph.graph["graph_id"] = graph_id
        degrees = [degree for _, degree in graph.degree()]
        graph_rows.append(
            {
                "graph_id": graph_id,
                "graph_seed": graph_seed,
                "n": graph.number_of_nodes(),
                "m": config.graph_m,
                "mean_degree": float(sum(degrees) / len(degrees)),
                "max_degree": int(max(degrees)),
            }
        )
        graphs.append(graph)

    return graphs, pd.DataFrame(graph_rows)


def clone_state(state: CascadeState) -> CascadeState:
    return state.copy()


def count_post_cascade_failures(state: CascadeState) -> int:
    preview = clone_state(state)
    if preview.frontier and preview.failed:
        advance_cascade_round(preview)
    return len(preview.failed)


def rollout_from_observation(
    env: RecoveryEnv,
    observation: RecoveryObservation,
    policy: Any,
) -> float:
    current_anc = env.current_anc()
    if not observation.failed or observation.remaining_budget <= 0:
        return current_anc

    done = False
    final_anc = current_anc
    while not done:
        action = policy(observation)
        observation, _reward, done, info = env.step(action)
        final_anc = float(info["anc"])
    return final_anc


def classify_instance(
    anc_degree: float,
    anc_random: float,
    spread: float,
    *,
    delta_h: float,
    delta_t: float,
    delta_s: float,
) -> str:
    if anc_degree < delta_h:
        return "hopeless"
    if anc_random > delta_t:
        return "trivial"
    if spread >= delta_s:
        return "decision_sensitive"
    return "ambiguous"


def classify_instances(
    frame: pd.DataFrame,
    *,
    delta_h: float,
    delta_t: float,
    delta_s: float,
) -> pd.Series:
    labels = pd.Series("ambiguous", index=frame.index, dtype="object")
    labels = labels.mask(frame["spread"] >= delta_s, "decision_sensitive")
    labels = labels.mask(frame["anc_random"] > delta_t, "trivial")
    labels = labels.mask(frame["anc_degree"] < delta_h, "hopeless")
    return labels


def evaluate_instance(
    graph: Any,
    graph_row: dict[str, Any],
    *,
    alpha: float,
    pfail: float,
    budget_ref: int,
    seed: int,
    config: MappingConfig,
    policy_factories: dict[str, Any],
) -> dict[str, Any]:
    scaled_budget = compute_scaled_budget(
        budget_ref,
        num_nodes=int(graph_row["n"]),
        reference_n=config.reference_n,
        enabled=True,
    )
    env_kwargs = {
        "alpha": alpha,
        "pfail": pfail,
        "budget": scaled_budget,
        "max_rounds": config.max_rounds,
    }
    env_degree = RecoveryEnv(graph, **env_kwargs, seed=seed)
    env_random = RecoveryEnv(graph, **env_kwargs, seed=seed)

    degree_observation = env_degree.reset(seed=seed)
    random_observation = env_random.reset(seed=seed)

    degree_failed = frozenset(degree_observation.failed)
    random_failed = frozenset(random_observation.failed)
    assert degree_failed == random_failed, "Degree and random policies must share the same initial failures."
    assert degree_observation.frontier == random_observation.frontier, "Frontier mismatch for matched seed."

    n_initial_failures = len(degree_failed)
    n_post_cascade_failures_degree = count_post_cascade_failures(env_degree.state)
    n_post_cascade_failures_random = count_post_cascade_failures(env_random.state)
    assert n_post_cascade_failures_degree == n_post_cascade_failures_random, (
        "Degree and random policies must share the same t=0 post-cascade failure count."
    )

    graph_id = int(graph_row["graph_id"])
    degree_policy = policy_factories["degree"](graph_id, seed)
    random_policy = policy_factories["random"](graph_id, seed)

    anc_degree = rollout_from_observation(env_degree, degree_observation, degree_policy)
    anc_random = rollout_from_observation(env_random, random_observation, random_policy)
    spread = anc_degree - anc_random
    feasibility_ratio = (
        n_initial_failures / float(scaled_budget * config.max_rounds)
        if scaled_budget * config.max_rounds > 0
        else 0.0
    )
    instance_label = classify_instance(
        anc_degree,
        anc_random,
        spread,
        delta_h=config.delta_h,
        delta_t=config.delta_t,
        delta_s=config.delta_s,
    )

    return {
        "graph_id": graph_id,
        "graph_seed": int(graph_row["graph_seed"]),
        "n": int(graph_row["n"]),
        "m": int(graph_row["m"]),
        "mean_degree": float(graph_row["mean_degree"]),
        "max_degree": int(graph_row["max_degree"]),
        "alpha": float(alpha),
        "pfail": float(pfail),
        "budget_ref": int(budget_ref),
        "scaled_budget": int(scaled_budget),
        "seed": int(seed),
        "anc_degree": float(anc_degree),
        "anc_random": float(anc_random),
        "spread": float(spread),
        "n_initial_failures": int(n_initial_failures),
        "n_post_cascade_failures": int(n_post_cascade_failures_degree),
        "instance_label": instance_label,
        "feasibility_ratio": float(feasibility_ratio),
    }


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.parquet"


def read_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=INSTANCE_COLUMNS)
    frame = pd.read_parquet(path)
    return frame[INSTANCE_COLUMNS] if not frame.empty else pd.DataFrame(columns=INSTANCE_COLUMNS)


def completed_cells(frame: pd.DataFrame) -> set[tuple[float, float, int]]:
    if frame.empty:
        return set()
    cells = frame[["alpha", "pfail", "budget_ref"]].drop_duplicates()
    return {
        (float(row.alpha), float(row.pfail), int(row.budget_ref))
        for row in cells.itertuples(index=False)
    }


def write_parquet_frame(frame: pd.DataFrame, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(parquet_metadata(metadata))
    pq.write_table(table, path)


def write_csv_frame(frame: pd.DataFrame, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(f"# generated_by: {metadata['generated_by']}\n")
        file.write(f"# master_seed: {metadata['master_seed']}\n")
        file.write(f"# timestamp: {metadata['timestamp']}\n")
        file.write(f"# grid: {json.dumps(metadata['grid'], sort_keys=True)}\n")
        frame.to_csv(file, index=False)


def write_json_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def validate_same_graph_invariant(instance_frame: pd.DataFrame) -> None:
    if instance_frame.empty:
        return
    n_counts = instance_frame.groupby("graph_id")["n"].nunique()
    graph_seed_counts = instance_frame.groupby("graph_id")["graph_seed"].nunique()
    m_counts = instance_frame.groupby("graph_id")["m"].nunique()
    assert bool((n_counts == 1).all()), "Node counts changed across cells for the same graph_id."
    assert bool((graph_seed_counts == 1).all()), "Graph seeds changed across cells for the same graph_id."
    assert bool((m_counts == 1).all()), "BA attachment parameter changed across cells for the same graph_id."


def map_instances(
    config: MappingConfig,
    graphs: list[Any],
    graph_frame: pd.DataFrame,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(output_dir)
    checkpoint_frame = read_checkpoint(checkpoint)
    validate_same_graph_invariant(checkpoint_frame)
    done_cells = completed_cells(checkpoint_frame)
    graph_rows = {
        int(row.graph_id): row._asdict()
        for row in graph_frame.itertuples(index=False)
    }
    policy_factories = build_policy_factories(base_seed=config.master_seed)

    all_cells = list(product(config.alpha_values, config.pfail_values, config.budget_values))
    if done_cells:
        print(f"Resuming from checkpoint with {len(done_cells)} completed cells.")

    for alpha, pfail, budget_ref in all_cells:
        cell_key = (float(alpha), float(pfail), int(budget_ref))
        if cell_key in done_cells:
            print(
                f"Skipping completed cell alpha={alpha:.2f}, pfail={pfail:.2f}, budget={budget_ref}."
            )
            continue

        rows: list[dict[str, Any]] = []
        total_steps = config.n_graphs * config.n_seeds * 2
        with tqdm(
            total=total_steps,
            desc=f"Mapping [alpha={alpha:.2f} pfail={pfail:.2f} budget={budget_ref}]",
            bar_format="{desc}: {percentage:3.0f}% |{bar}| ETA: {remaining}",
        ) as progress:
            for graph_id, graph in enumerate(graphs):
                graph_row = graph_rows[graph_id]
                assert int(graph_row["n"]) == graph.number_of_nodes(), "Graph node count changed for graph_id."
                for seed in config.seeds:
                    row = evaluate_instance(
                        graph,
                        graph_row,
                        alpha=alpha,
                        pfail=pfail,
                        budget_ref=budget_ref,
                        seed=seed,
                        config=config,
                        policy_factories=policy_factories,
                    )
                    rows.append(row)
                    progress.update(2)

        cell_frame = pd.DataFrame(rows, columns=INSTANCE_COLUMNS)
        if checkpoint_frame.empty:
            checkpoint_frame = cell_frame.copy()
        else:
            checkpoint_frame = pd.concat([checkpoint_frame, cell_frame], ignore_index=True)
        checkpoint_frame = checkpoint_frame.sort_values(
            by=["alpha", "pfail", "budget_ref", "graph_id", "seed"]
        ).reset_index(drop=True)
        validate_same_graph_invariant(checkpoint_frame)
        write_parquet_frame(checkpoint_frame, checkpoint, metadata)
        done_cells.add(cell_key)

    return checkpoint_frame[INSTANCE_COLUMNS].copy()


def aggregate_cells_generic(
    frame: pd.DataFrame,
    *,
    label_column: str,
    min_ds_frac: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["alpha", "pfail", "budget_ref"], sort=True)

    for (alpha, pfail, budget_ref), group in grouped:
        counts = group[label_column].value_counts(normalize=True)
        f_hopeless = float(counts.get("hopeless", 0.0))
        f_trivial = float(counts.get("trivial", 0.0))
        f_ds = float(counts.get("decision_sensitive", 0.0))
        f_ambiguous = float(counts.get("ambiguous", 0.0))

        if f_hopeless > 0.50:
            cell_label = "hopeless"
        elif f_trivial > 0.50:
            cell_label = "trivial"
        elif f_ds >= min_ds_frac:
            cell_label = "decision_sensitive"
        else:
            cell_label = "mixed"

        ds_group = group[group[label_column] == "decision_sensitive"]
        spread_mean_of_ds_instances = float(ds_group["spread"].mean()) if not ds_group.empty else 0.0
        interestingness = float(f_ds * spread_mean_of_ds_instances)

        rows.append(
            {
                "alpha": float(alpha),
                "pfail": float(pfail),
                "budget_ref": int(budget_ref),
                "n_instances": int(len(group)),
                "anc_degree_mean": float(group["anc_degree"].mean()),
                "anc_degree_std": safe_std(group["anc_degree"]),
                "anc_random_mean": float(group["anc_random"].mean()),
                "anc_random_std": safe_std(group["anc_random"]),
                "spread_mean": float(group["spread"].mean()),
                "spread_std": safe_std(group["spread"]),
                "spread_p25": float(group["spread"].quantile(0.25)),
                "spread_p75": float(group["spread"].quantile(0.75)),
                "f_hopeless": f_hopeless,
                "f_trivial": f_trivial,
                "f_ds": f_ds,
                "f_ambiguous": f_ambiguous,
                "cell_label": cell_label,
                "interestingness": interestingness,
                "spread_mean_of_ds_instances": spread_mean_of_ds_instances,
                "feasibility_ratio_mean": float(group["feasibility_ratio"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["budget_ref", "alpha", "pfail"]).reset_index(drop=True)


def aggregate_budget_summary(cell_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = cell_frame.groupby("budget_ref", sort=True)
    for budget_ref, group in grouped:
        rows.append(
            {
                "budget_ref": int(budget_ref),
                "n_cells_ds": int((group["cell_label"] == "decision_sensitive").sum()),
                "mean_f_ds": float(group["f_ds"].mean()),
                "mean_interestingness": float(group["interestingness"].mean()),
                "std_interestingness": safe_std(group["interestingness"]),
                "feasibility_ratio_mean": float(group["feasibility_ratio_mean"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(by="budget_ref").reset_index(drop=True)


def sensitivity_entry_lookup(
    sensitivity_frame: pd.DataFrame,
    *,
    delta_h: float,
    delta_t: float,
    delta_s: float,
    min_ds_frac: float,
) -> pd.Series:
    match = sensitivity_frame[
        (sensitivity_frame["delta_h"] == delta_h)
        & (sensitivity_frame["delta_t"] == delta_t)
        & (sensitivity_frame["delta_s"] == delta_s)
        & (sensitivity_frame["min_ds_frac"] == min_ds_frac)
    ]
    if len(match) != 1:
        raise AssertionError("Expected exactly one threshold sensitivity entry for the default thresholds.")
    return match.iloc[0]


def compute_stable_zone(
    sensitivity_frame: pd.DataFrame,
    config: MappingConfig,
) -> list[dict[str, Any]]:
    delta_h_index = {value: idx for idx, value in enumerate(config.delta_h_candidates)}
    delta_t_index = {value: idx for idx, value in enumerate(config.delta_t_candidates)}
    delta_s_index = {value: idx for idx, value in enumerate(config.delta_s_candidates)}
    min_ds_frac_index = {
        value: idx for idx, value in enumerate(config.min_ds_frac_candidates)
    }
    lookup = {
        (row.delta_h, row.delta_t, row.delta_s, row.min_ds_frac): row
        for row in sensitivity_frame.itertuples(index=False)
    }
    stable_entries: list[dict[str, Any]] = []

    for key, row in lookup.items():
        h, t, s, m = key
        h_idx = delta_h_index[h]
        t_idx = delta_t_index[t]
        s_idx = delta_s_index[s]
        m_idx = min_ds_frac_index[m]
        neighbor_keys: list[tuple[float, float, float, float]] = []

        for delta in (-1, 1):
            if 0 <= h_idx + delta < len(config.delta_h_candidates):
                neighbor_keys.append(
                    (
                        config.delta_h_candidates[h_idx + delta],
                        t,
                        s,
                        m,
                    )
                )
            if 0 <= t_idx + delta < len(config.delta_t_candidates):
                neighbor_keys.append(
                    (
                        h,
                        config.delta_t_candidates[t_idx + delta],
                        s,
                        m,
                    )
                )
            if 0 <= s_idx + delta < len(config.delta_s_candidates):
                neighbor_keys.append(
                    (
                        h,
                        t,
                        config.delta_s_candidates[s_idx + delta],
                        m,
                    )
                )
            if 0 <= m_idx + delta < len(config.min_ds_frac_candidates):
                neighbor_keys.append(
                    (
                        h,
                        t,
                        s,
                        config.min_ds_frac_candidates[m_idx + delta],
                    )
                )

        neighbor_changes = [
            abs(int(row.n_cells_ds) - int(lookup[neighbor_key].n_cells_ds))
            for neighbor_key in neighbor_keys
        ]
        max_adjacent_ds_change = max(neighbor_changes) if neighbor_changes else 0
        if max_adjacent_ds_change < 2:
            stable_entries.append(
                {
                    "delta_h": float(h),
                    "delta_t": float(t),
                    "delta_s": float(s),
                    "min_ds_frac": float(m),
                    "n_cells_ds": int(row.n_cells_ds),
                    "mean_f_ds": float(row.mean_f_ds),
                    "mean_interestingness": float(row.mean_interestingness),
                    "max_adjacent_ds_change": int(max_adjacent_ds_change),
                }
            )

    stable_entries.sort(
        key=lambda entry: (
            -entry["mean_interestingness"],
            -entry["mean_f_ds"],
            -entry["n_cells_ds"],
            entry["delta_h"],
            entry["delta_t"],
            entry["delta_s"],
            entry["min_ds_frac"],
        )
    )
    return stable_entries


def run_threshold_sensitivity(
    instance_frame: pd.DataFrame,
    cell_frame: pd.DataFrame,
    config: MappingConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for delta_h, delta_t, delta_s, min_ds_frac in product(
        config.delta_h_candidates,
        config.delta_t_candidates,
        config.delta_s_candidates,
        config.min_ds_frac_candidates,
    ):
        temp = instance_frame[
            ["alpha", "pfail", "budget_ref", "anc_degree", "anc_random", "spread", "feasibility_ratio"]
        ].copy()
        temp["candidate_label"] = classify_instances(
            temp,
            delta_h=delta_h,
            delta_t=delta_t,
            delta_s=delta_s,
        )
        temp_cells = aggregate_cells_generic(
            temp.rename(columns={"candidate_label": "instance_label"}),
            label_column="instance_label",
            min_ds_frac=min_ds_frac,
        )
        label_counts = temp_cells["cell_label"].value_counts()
        rows.append(
            {
                "delta_h": float(delta_h),
                "delta_t": float(delta_t),
                "delta_s": float(delta_s),
                "min_ds_frac": float(min_ds_frac),
                "n_cells_decision_sensitive": int(label_counts.get("decision_sensitive", 0)),
                "n_cells_hopeless": int(label_counts.get("hopeless", 0)),
                "n_cells_trivial": int(label_counts.get("trivial", 0)),
                "n_cells_mixed": int(label_counts.get("mixed", 0)),
                "mean_f_ds": float(temp_cells["f_ds"].mean()),
                "mean_interestingness": float(temp_cells["interestingness"].mean()),
            }
        )

    sensitivity_frame = pd.DataFrame(rows).rename(
        columns={"n_cells_decision_sensitive": "n_cells_ds"}
    )
    default_entry = sensitivity_entry_lookup(
        sensitivity_frame,
        delta_h=config.delta_h,
        delta_t=config.delta_t,
        delta_s=config.delta_s,
        min_ds_frac=config.min_ds_frac,
    )
    main_label_counts = cell_frame["cell_label"].value_counts()
    assert int(default_entry["n_cells_ds"]) == int(main_label_counts.get("decision_sensitive", 0))
    assert int(default_entry["n_cells_hopeless"]) == int(main_label_counts.get("hopeless", 0))
    assert int(default_entry["n_cells_trivial"]) == int(main_label_counts.get("trivial", 0))
    assert int(default_entry["n_cells_mixed"]) == int(main_label_counts.get("mixed", 0))
    stable_zone = compute_stable_zone(sensitivity_frame, config)
    return sensitivity_frame, stable_zone


def save_plot(fig: Any, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata=png_metadata(metadata))
    plt.close(fig)


def build_matrix(
    cell_frame: pd.DataFrame,
    *,
    budget_ref: int,
    value_column: str,
    alphas: tuple[float, ...],
    pfails: tuple[float, ...],
) -> list[list[float]]:
    subset = cell_frame[cell_frame["budget_ref"] == budget_ref]
    matrix: list[list[float]] = []
    for pfail in pfails:
        row: list[float] = []
        for alpha in alphas:
            match = subset[(subset["alpha"] == alpha) & (subset["pfail"] == pfail)]
            row.append(float(match.iloc[0][value_column]) if not match.empty else math.nan)
        matrix.append(row)
    return matrix


def plot_spread_histogram_by_alpha(
    instance_frame: pd.DataFrame,
    config: MappingConfig,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for axis, alpha in zip(axes.flat, config.alpha_values, strict=False):
        subset = instance_frame[instance_frame["alpha"] == alpha]["spread"]
        axis.hist(subset, bins=25, color="steelblue", alpha=0.85, label="spread")
        axis.axvline(config.delta_s, color="red", linestyle="--", label=f"DELTA_S={config.delta_s:.2f}")
        axis.axvline(float(subset.quantile(0.25)), color="grey", linestyle=":", label="p25/p75")
        axis.axvline(float(subset.quantile(0.75)), color="grey", linestyle=":")
        axis.set_title(f"alpha={alpha:.2f} (n={len(subset)})")
        axis.set_xlabel("Spread (degree - random)")
        axis.set_ylabel("Count")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Spread Histogram By Alpha")
    save_plot(fig, output_dir / "plots" / "spread_histogram_by_alpha.png", metadata)


def plot_anc_degree_histogram_by_alpha(
    instance_frame: pd.DataFrame,
    config: MappingConfig,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for axis, alpha in zip(axes.flat, config.alpha_values, strict=False):
        subset = instance_frame[instance_frame["alpha"] == alpha]["anc_degree"]
        axis.hist(subset, bins=25, color="darkgreen", alpha=0.85, label="anc_degree")
        axis.axvline(config.delta_h, color="red", linestyle="--", label=f"DELTA_H={config.delta_h:.2f}")
        axis.axvline(config.delta_t, color="blue", linestyle="--", label=f"DELTA_T={config.delta_t:.2f}")
        axis.set_title(f"alpha={alpha:.2f} (n={len(subset)})")
        axis.set_xlabel("Degree-policy final ANC")
        axis.set_ylabel("Count")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Degree ANC Histogram By Alpha")
    save_plot(fig, output_dir / "plots" / "anc_degree_histogram_by_alpha.png", metadata)


def annotate_heatmap(
    axis: Any,
    matrix: list[list[float]],
    *,
    alphas: tuple[float, ...],
    pfails: tuple[float, ...],
    top_cells: set[tuple[float, float]] | None = None,
) -> None:
    for row_idx, pfail in enumerate(pfails):
        for col_idx, alpha in enumerate(alphas):
            value = matrix[row_idx][col_idx]
            if math.isnan(value):
                text = "NA"
            else:
                starred = top_cells is not None and (alpha, pfail) in top_cells
                text = f"{value:.2f}{'*' if starred else ''}"
            axis.text(col_idx, row_idx, text, ha="center", va="center", color="black")


def plot_decision_sensitive_fraction_heatmap(
    cell_frame: pd.DataFrame,
    config: MappingConfig,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=config.min_ds_frac, vmax=1.0)
    image = None
    for axis, budget_ref in zip(axes.flat, config.budget_values, strict=False):
        matrix = build_matrix(
            cell_frame,
            budget_ref=budget_ref,
            value_column="f_ds",
            alphas=config.alpha_values,
            pfails=config.pfail_values,
        )
        image = axis.imshow(matrix, cmap="coolwarm", norm=norm, aspect="auto")
        annotate_heatmap(axis, matrix, alphas=config.alpha_values, pfails=config.pfail_values)
        axis.set_xticks(range(len(config.alpha_values)), [f"{alpha:.2f}" for alpha in config.alpha_values])
        axis.set_yticks(range(len(config.pfail_values)), [f"{pfail:.2f}" for pfail in config.pfail_values])
        axis.set_xlabel("alpha")
        axis.set_ylabel("pfail")
        axis.set_title(f"budget_ref={budget_ref}")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="f_ds")
    fig.suptitle("Decision-Sensitive Fraction Heatmap")
    save_plot(fig, output_dir / "plots" / "decision_sensitive_fraction_heatmap.png", metadata)


def plot_interestingness_heatmap(
    cell_frame: pd.DataFrame,
    config: MappingConfig,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    vmax = float(cell_frame["interestingness"].max()) if not cell_frame.empty else 1.0
    image = None
    for axis, budget_ref in zip(axes.flat, config.budget_values, strict=False):
        subset = cell_frame[cell_frame["budget_ref"] == budget_ref]
        matrix = build_matrix(
            cell_frame,
            budget_ref=budget_ref,
            value_column="interestingness",
            alphas=config.alpha_values,
            pfails=config.pfail_values,
        )
        top_cells = {
            (float(row.alpha), float(row.pfail))
            for row in subset.nlargest(3, "interestingness").itertuples(index=False)
        }
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-9), aspect="auto")
        annotate_heatmap(
            axis,
            matrix,
            alphas=config.alpha_values,
            pfails=config.pfail_values,
            top_cells=top_cells,
        )
        axis.set_xticks(range(len(config.alpha_values)), [f"{alpha:.2f}" for alpha in config.alpha_values])
        axis.set_yticks(range(len(config.pfail_values)), [f"{pfail:.2f}" for pfail in config.pfail_values])
        axis.set_xlabel("alpha")
        axis.set_ylabel("pfail")
        axis.set_title(f"budget_ref={budget_ref}")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="interestingness")
    fig.suptitle("Interestingness Heatmap")
    save_plot(fig, output_dir / "plots" / "interestingness_heatmap.png", metadata)


def plot_budget_comparison_barplot(
    budget_frame: pd.DataFrame,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(12, 7))
    x_positions = list(range(len(budget_frame)))
    width = 0.23
    axis.bar(
        [value - width for value in x_positions],
        budget_frame["n_cells_ds"],
        width=width,
        label="n_cells_ds",
    )
    axis.bar(
        x_positions,
        budget_frame["mean_f_ds"] * 100.0,
        width=width,
        label="mean_f_ds (%)",
    )
    axis.bar(
        [value + width for value in x_positions],
        budget_frame["mean_interestingness"] * 100.0,
        width=width,
        label="mean_interestingness (%)",
    )
    axis.set_xticks(x_positions, [str(int(value)) for value in budget_frame["budget_ref"]])
    axis.set_xlabel("budget_ref")
    axis.set_ylabel("Cell count / percentage")
    axis.set_title("Budget Comparison")
    axis.legend(loc="upper left")

    second_axis = axis.twinx()
    second_axis.plot(
        x_positions,
        budget_frame["feasibility_ratio_mean"],
        color="black",
        marker="o",
        linewidth=2,
        label="feasibility_ratio_mean",
    )
    second_axis.set_ylabel("Feasibility ratio mean")
    second_axis.legend(loc="upper right")
    save_plot(fig, output_dir / "plots" / "budget_comparison_barplot.png", metadata)


def make_plots(
    instance_frame: pd.DataFrame,
    cell_frame: pd.DataFrame,
    budget_frame: pd.DataFrame,
    *,
    output_dir: Path,
    config: MappingConfig,
    metadata: dict[str, Any],
) -> None:
    plot_spread_histogram_by_alpha(instance_frame, config, output_dir=output_dir, metadata=metadata)
    plot_anc_degree_histogram_by_alpha(
        instance_frame,
        config,
        output_dir=output_dir,
        metadata=metadata,
    )
    plot_decision_sensitive_fraction_heatmap(
        cell_frame,
        config,
        output_dir=output_dir,
        metadata=metadata,
    )
    plot_interestingness_heatmap(
        cell_frame,
        config,
        output_dir=output_dir,
        metadata=metadata,
    )
    plot_budget_comparison_barplot(budget_frame, output_dir=output_dir, metadata=metadata)


def choose_recommended_cell(cell_frame: pd.DataFrame) -> dict[str, Any] | None:
    if cell_frame.empty:
        return None
    candidate_frame = cell_frame[cell_frame["cell_label"] == "decision_sensitive"]
    if candidate_frame.empty:
        candidate_frame = cell_frame
    best_row = candidate_frame.sort_values(
        by=["interestingness", "f_ds", "spread_mean"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "alpha": float(best_row["alpha"]),
        "pfail": float(best_row["pfail"]),
        "budget_ref": int(best_row["budget_ref"]),
        "interestingness": float(best_row["interestingness"]),
        "f_ds": float(best_row["f_ds"]),
        "cell_label": str(best_row["cell_label"]),
    }


def print_summary_table(
    budget_frame: pd.DataFrame,
    recommended_cell: dict[str, Any] | None,
) -> None:
    print("Budget | DS Cells | Mean f_DS | Mean Interest. | Feasibility")
    print("-------|----------|-----------|----------------|------------")
    for row in budget_frame.itertuples(index=False):
        print(
            f"{int(row.budget_ref):>6} |"
            f" {int(row.n_cells_ds):>8} |"
            f" {float(row.mean_f_ds):>9.3f} |"
            f" {float(row.mean_interestingness):>14.3f} |"
            f" {float(row.feasibility_ratio_mean):>10.3f}"
        )
    if recommended_cell is None:
        print("Recommended training cell: none available")
        return
    print(
        "Recommended training cell: "
        f"alpha={recommended_cell['alpha']:.2f}, "
        f"pfail={recommended_cell['pfail']:.2f}, "
        f"budget={recommended_cell['budget_ref']} "
        f"(interestingness={recommended_cell['interestingness']:.3f}, "
        f"f_ds={recommended_cell['f_ds']:.3f})"
    )


def write_outputs(
    *,
    instance_frame: pd.DataFrame,
    graph_frame: pd.DataFrame,
    cell_frame: pd.DataFrame,
    budget_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
    stable_zone: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    instance_parquet = output_dir / "regime_instances.parquet"
    instance_csv = output_dir / "regime_instances.csv"
    cells_json = output_dir / "regime_cells.json"
    cells_csv = output_dir / "regime_cells.csv"
    budget_json = output_dir / "budget_summary.json"
    sensitivity_json = output_dir / "threshold_sensitivity.json"

    write_parquet_frame(instance_frame, instance_parquet, metadata)
    write_csv_frame(instance_frame, instance_csv, metadata)
    write_csv_frame(cell_frame, cells_csv, metadata)

    recommended_cell = choose_recommended_cell(cell_frame)
    write_json_payload(
        {
            "metadata": metadata,
            "graphs": graph_frame.to_dict(orient="records"),
            "cells": cell_frame.to_dict(orient="records"),
            "recommended_training_cell": recommended_cell,
        },
        cells_json,
    )
    write_json_payload(
        {
            "metadata": metadata,
            "budgets": budget_frame.to_dict(orient="records"),
        },
        budget_json,
    )
    write_json_payload(
        {
            "metadata": metadata,
            "threshold_sensitivity": sensitivity_frame.to_dict(orient="records"),
            "stable_zone": stable_zone,
        },
        sensitivity_json,
    )


def run_analysis(
    config: MappingConfig | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config = config or default_config()
    resolved_output_dir = output_dir or default_output_dir(config)
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = build_metadata(config, timestamp)

    graphs, graph_frame = build_graph_bank(config)
    instance_frame = map_instances(
        config,
        graphs,
        graph_frame,
        output_dir=resolved_output_dir,
        metadata=metadata,
    )
    instance_frame = instance_frame.sort_values(
        by=["alpha", "pfail", "budget_ref", "graph_id", "seed"]
    ).reset_index(drop=True)
    validate_same_graph_invariant(instance_frame)

    cell_frame = aggregate_cells_generic(
        instance_frame,
        label_column="instance_label",
        min_ds_frac=config.min_ds_frac,
    )
    budget_frame = aggregate_budget_summary(cell_frame)
    sensitivity_frame, stable_zone = run_threshold_sensitivity(instance_frame, cell_frame, config)

    write_outputs(
        instance_frame=instance_frame,
        graph_frame=graph_frame,
        cell_frame=cell_frame,
        budget_frame=budget_frame,
        sensitivity_frame=sensitivity_frame,
        stable_zone=stable_zone,
        metadata=metadata,
        output_dir=resolved_output_dir,
    )
    make_plots(
        instance_frame,
        cell_frame,
        budget_frame,
        output_dir=resolved_output_dir,
        config=config,
        metadata=metadata,
    )

    print_summary_table(budget_frame, choose_recommended_cell(cell_frame))
    if stable_zone:
        print(
            "Stable zone threshold combinations "
            f"({len(stable_zone)} total; showing up to 5):"
        )
        for entry in stable_zone[:5]:
            print(
                f"  delta_h={entry['delta_h']:.2f}, delta_t={entry['delta_t']:.2f}, "
                f"delta_s={entry['delta_s']:.2f}, min_ds_frac={entry['min_ds_frac']:.2f}, "
                f"n_cells_ds={entry['n_cells_ds']}"
            )

    return {
        "metadata": metadata,
        "graphs": graph_frame,
        "instances": instance_frame,
        "cells": cell_frame,
        "budget_summary": budget_frame,
        "threshold_sensitivity": sensitivity_frame,
        "stable_zone": stable_zone,
        "output_dir": resolved_output_dir,
    }


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
