from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def plot_interestingness_heatmaps(results: dict, output_path: Path) -> None:
    cells = results["cells"]
    budgets = sorted({cell["budget"] for cell in cells})
    columns = min(2, len(budgets))
    rows = math.ceil(len(budgets) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4.8 * rows), constrained_layout=True
    )
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]

    image = None
    for axis, budget in zip(axes_list, budgets):
        alphas, pfails, grid = _prepare_grid(cells, budget, "interestingness_score")
        image = axis.imshow(grid, aspect="auto", origin="lower")
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for policy_name in results["policies"]:
        budgets = [cell["budget"] for cell in cells]
        final_anc = [cell["policy_summaries"][policy_name]["final_anc"]["mean"] for cell in cells]
        threshold_hit = [
            cell["policy_summaries"][policy_name]["threshold_hit_fraction"]["mean"]
            for cell in cells
        ]
        axes[0].plot(budgets, final_anc, marker="o", label=policy_name)
        axes[1].plot(budgets, threshold_hit, marker="o", label=policy_name)

    axes[0].set_title(f"Final ANC vs budget (alpha={target_alpha}, pfail={target_pfail})")
    axes[0].set_xlabel("budget")
    axes[0].set_ylabel("mean final ANC")
    axes[1].set_title("Threshold-hit fraction vs budget")
    axes[1].set_xlabel("budget")
    axes[1].set_ylabel("mean threshold-hit fraction")
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


if __name__ == "__main__":
    main()
