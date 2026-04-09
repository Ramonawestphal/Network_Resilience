from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _prepare_grid(cells: list[dict], budget: int, value_key: str) -> tuple[list[float], list[float], np.ndarray]:
    budget_cells = [cell for cell in cells if cell["budget"] == budget]
    alphas = sorted({cell["alpha"] for cell in budget_cells})
    pfails = sorted({cell["pfail"] for cell in budget_cells})
    grid = np.full((len(alphas), len(pfails)), np.nan)

    alpha_index = {alpha: index for index, alpha in enumerate(alphas)}
    pfail_index = {pfail: index for index, pfail in enumerate(pfails)}
    for cell in budget_cells:
        grid[alpha_index[cell["alpha"]], pfail_index[cell["pfail"]]] = cell["diagnostics"][value_key]

    return alphas, pfails, grid


def _prepare_policy_metric_grid(
    cells: list[dict],
    budget: int,
    policy_name: str,
    value_fn: Callable[[dict, str], float],
) -> tuple[list[float], list[float], np.ndarray]:
    budget_cells = [cell for cell in cells if cell["budget"] == budget]
    alphas = sorted({cell["alpha"] for cell in budget_cells})
    pfails = sorted({cell["pfail"] for cell in budget_cells})
    grid = np.full((len(alphas), len(pfails)), np.nan)

    alpha_index = {alpha: index for index, alpha in enumerate(alphas)}
    pfail_index = {pfail: index for index, pfail in enumerate(pfails)}
    for cell in budget_cells:
        grid[alpha_index[cell["alpha"]], pfail_index[cell["pfail"]]] = value_fn(
            cell, policy_name
        )

    return alphas, pfails, grid


def plot_policy_metric_heatmaps(
    results: dict,
    output_path: Path,
    *,
    suptitle: str,
    colorbar_label: str,
    value_fn: Callable[[dict, str], float],
) -> None:
    """One row per budget, one column per policy: alpha×pfail heatmaps of ``value_fn(cell, policy)``."""
    cells = results["cells"]
    policies: list[str] = list(results["policies"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cells or not policies:
        logger.warning(
            "plot_policy_metric_heatmaps: empty cells or policies; writing placeholder to %s",
            output_path,
        )
        fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        axis.text(
            0.5,
            0.5,
            "No regime cells to plot",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=12,
        )
        axis.set_axis_off()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    budgets = sorted({cell["budget"] for cell in cells})
    rows, cols = len(budgets), len(policies)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.6 * rows), constrained_layout=True
    )
    if rows == 1 and cols == 1:
        axes_array = np.array([[axes]])
    elif rows == 1:
        axes_array = np.array([axes])
    elif cols == 1:
        axes_array = np.array([[ax] for ax in axes])
    else:
        axes_array = np.array(axes)

    stacked_vals: list[float] = []
    for bi, budget in enumerate(budgets):
        for pi, policy_name in enumerate(policies):
            _, _, grid = _prepare_policy_metric_grid(
                cells, budget, policy_name, value_fn
            )
            stacked_vals.extend(grid[np.isfinite(grid)].tolist())

    if stacked_vals:
        vmin = float(min(stacked_vals))
        vmax = float(max(stacked_vals))
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9

    image = None
    for bi, budget in enumerate(budgets):
        for pi, policy_name in enumerate(policies):
            ax = axes_array[bi, pi]
            alphas, pfails, grid = _prepare_policy_metric_grid(
                cells, budget, policy_name, value_fn
            )
            im = ax.imshow(grid, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            if image is None:
                image = im
            ax.set_title(f"{policy_name} (B={budget})")
            ax.set_xticks(range(len(pfails)), labels=[f"{p:.2f}" for p in pfails])
            ax.set_yticks(range(len(alphas)), labels=[f"{a:.2f}" for a in alphas])
            ax.set_xlabel("pfail")
            ax.set_ylabel("alpha")

    fig.suptitle(suptitle, fontsize=14)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes_array.ravel().tolist(),
            shrink=0.82,
            label=colorbar_label,
        )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_interestingness_heatmaps(results: dict, output_path: Path) -> None:
    cells = results["cells"]
    if not cells:
        logger.warning(
            "plot_interestingness_heatmaps: empty cells; writing placeholder to %s",
            output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        axis.text(
            0.5,
            0.5,
            "No regime cells to plot",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=12,
        )
        axis.set_axis_off()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    budgets = sorted({cell["budget"] for cell in cells})
    columns = min(2, len(budgets))
    rows = math.ceil(len(budgets) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4.8 * rows), constrained_layout=True
    )
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]

    prepared = [
        _prepare_grid(cells, budget, "interestingness_score") for budget in budgets
    ]
    stacked = np.concatenate([grid.ravel() for _, _, grid in prepared])
    finite = stacked[np.isfinite(stacked)]
    if finite.size:
        vmin = float(finite.min())
        vmax = float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9

    image = None
    for axis, budget, (alphas, pfails, grid) in zip(axes_list, budgets, prepared):
        im = axis.imshow(grid, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        if image is None:
            image = im
        axis.set_title(f"Interestingness score (budget={budget})")
        axis.set_xticks(range(len(pfails)), labels=[f"{pfail:.2f}" for pfail in pfails])
        axis.set_yticks(range(len(alphas)), labels=[f"{alpha:.2f}" for alpha in alphas])
        axis.set_xlabel("pfail")
        axis.set_ylabel("alpha")

    for axis in axes_list[len(budgets):]:
        axis.set_axis_off()

    if image is not None:
        fig.colorbar(image, ax=axes_list[: len(budgets)], shrink=0.85, label="score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _rounds_when_solved_mean(cell: dict, policy_name: str) -> float:
    rws = cell["policy_summaries"][policy_name].get("rounds_when_solved")
    if not isinstance(rws, dict):
        return float("nan")
    mean = rws.get("mean")
    if mean is None:
        return float("nan")
    m = float(mean)
    return m if math.isfinite(m) else float("nan")


def plot_budget_curves(results: dict, output_path: Path) -> None:
    recommendation = results["recommendation"]
    if not recommendation:
        return

    target_alpha = recommendation["alpha"]
    target_pfail = recommendation["pfail"]
    cells = [
        cell
        for cell in results["cells"]
        if cell["alpha"] == target_alpha and cell["pfail"] == target_pfail
    ]
    cells.sort(key=lambda cell: cell["budget"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for policy_name in results["policies"]:
        budgets = [cell["budget"] for cell in cells]
        final_anc = [cell["policy_summaries"][policy_name]["final_nc"]["mean"] for cell in cells]
        solved_frac = [
            cell["policy_summaries"][policy_name]["solved_fraction"]["mean"]
            for cell in cells
        ]
        rws_means = [_rounds_when_solved_mean(cell, policy_name) for cell in cells]
        axes[0].plot(budgets, final_anc, marker="o", label=policy_name)
        axes[1].plot(budgets, solved_frac, marker="o", label=policy_name)
        axes[2].plot(budgets, rws_means, marker="o", label=policy_name)

    axes[0].set_title(f"Final ANC vs budget (alpha={target_alpha}, pfail={target_pfail})")
    axes[0].set_xlabel("budget")
    axes[0].set_ylabel("mean final ANC")
    axes[1].set_title("Fully restored fraction vs budget")
    axes[1].set_xlabel("budget")
    axes[1].set_ylabel("mean solved fraction")
    axes[2].set_title("Rounds when fully restored vs budget")
    axes[2].set_xlabel("budget")
    axes[2].set_ylabel("mean rounds (solved episodes only)")
    axes[0].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved regime-mapping results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "experiments" / "regime_map" / "regime_results.json",
        help="Path to a saved regime-results JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Defaults to the results file directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.results)
    output_dir = args.output_dir or args.results.parent
    plot_interestingness_heatmaps(results, output_dir / "interestingness_heatmap.png")
    plot_budget_curves(results, output_dir / "budget_curves.png")
    plot_policy_metric_heatmaps(
        results,
        output_dir / "solved_fraction_heatmaps.png",
        suptitle="Mean solved fraction (heuristic rollouts)",
        colorbar_label="solved fraction",
        value_fn=lambda cell, pol: float(
            cell["policy_summaries"][pol]["solved_fraction"]["mean"]
        ),
    )
    plot_policy_metric_heatmaps(
        results,
        output_dir / "unsolved_low_anc_heatmaps.png",
        suptitle="Unsolved low-final-ANC fraction (failed & final ANC < threshold)",
        colorbar_label="fraction of episodes",
        value_fn=lambda cell, pol: (
            float(v)
            if (v := cell["policy_summaries"][pol].get("unsolved_low_final_nc_fraction"))
            is not None
            else float("nan")
        ),
    )


if __name__ == "__main__":
    main()
