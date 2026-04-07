"""Plot evaluation results from evaluation_summary.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

POLICY_ORDER = ["rl", "greedy", "degree", "betweenness", "risk", "random"]
POLICY_COLORS = {
    "rl": "#2196F3",
    "greedy": "#4CAF50",
    "degree": "#FF9800",
    "betweenness": "#9C27B0",
    "risk": "#F44336",
    "random": "#9E9E9E",
}


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_policies(data: dict) -> list[str]:
    skip = {"checkpoint", "config", "policies", "minimum_budget_solved_target",
            "env_stopping", "scaling"}
    found = [p for p in POLICY_ORDER if p in data and p not in skip]
    extras = [p for p in data if p not in found and p not in skip]
    return found + extras


def _mean_err(data: dict, policy: str, key: str) -> tuple[float, float]:
    """Get mean and stderr for a metric, handling old and new formats."""
    p = data.get(policy, {})
    if key in p and isinstance(p[key], dict):
        return float(p[key]["mean"]), float(p[key].get("stderr", 0.0))
    # old flat format fallbacks
    flat_map = {
        "final_nc": ("final_anc_mean", "final_anc_stderr"),
        "solved_fraction": ("solved_fraction_mean", None),
        "rounds": ("rounds_mean", None),
    }
    if key in flat_map:
        mk, ek = flat_map[key]
        mean = float(p.get(mk, 0.0))
        err = float(p.get(ek, 0.0)) if ek else 0.0
        return mean, err
    return 0.0, 0.0


def _scalar(data: dict, policy: str, key: str) -> float | None:
    p = data.get(policy, {})
    val = p.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        return float(val["mean"])
    return float(val)


def bar_chart(ax, policies, values, errors, title, ylabel, colors):
    x = np.arange(len(policies))
    bars = ax.bar(x, values, yerr=errors, capsize=4,
                  color=[colors.get(p, "#607D8B") for p in policies],
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return bars


def plot_main_metrics(data: dict, policies: list[str], out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Policy Evaluation — Main Metrics", fontsize=13, fontweight="bold")

    metrics = [
        ("final_nc",        "Final NC (snapshot)",         "NC"),
        ("anc_fixed",       "ANC Fixed Horizon",           "ANC"),
        ("anc_adaptive",    "ANC Adaptive Horizon",        "ANC"),
        ("solved_fraction", "Solved Fraction",             "Fraction"),
        ("rounds",          "Mean Rounds Taken",           "Rounds"),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        vals = [_mean_err(data, p, key)[0] for p in policies]
        errs = [_mean_err(data, p, key)[1] for p in policies]
        bar_chart(ax, policies, vals, errs, title, ylabel, POLICY_COLORS)

    # b_star in last panel
    ax = axes.flat[5]
    b_stars = [_scalar(data, p, "b_star") for p in policies]
    vals = [b if b is not None else 0 for b in b_stars]
    labels = [str(int(b)) if b is not None else "N/A" for b in b_stars]
    bars = bar_chart(ax, policies, vals, [0]*len(policies),
                     "Minimum Budget (b*)", "Budget", POLICY_COLORS)
    for bar, label in zip(bars, labels):
        if label == "N/A":
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "N/A",
                    ha="center", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()
    path = out_dir / "main_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_decision_quality(data: dict, policies: list[str], out_dir: Path):
    new_metrics = [
        ("mean_degree_ratio",    "Degree Ratio\n(chosen / max failed degree)", "Ratio"),
        ("mean_overload_risk",   "Overload Risk\n(max load/capacity of neighbors)", "Risk"),
        ("mean_nc_gain",         "NC Gain of Chosen Action",  "ΔNC"),
        ("mean_greedy_nc_gain",  "NC Gain of Greedy Oracle",  "ΔNC"),
        ("mean_action_rank",     "Action Rank\n(1 = optimal)", "Rank"),
    ]

    has_new = any(
        isinstance(data.get(policies[0], {}).get(key), dict)
        for key, _, _ in new_metrics
    )
    if not has_new:
        print("Decision-quality metrics not in summary (re-run evaluation to generate them).")
        return

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Policy Decision Quality", fontsize=13, fontweight="bold")

    for ax, (key, title, ylabel) in zip(axes, new_metrics):
        vals = [_mean_err(data, p, key)[0] for p in policies]
        errs = [_mean_err(data, p, key)[1] for p in policies]
        bar_chart(ax, policies, vals, errs, title, ylabel, POLICY_COLORS)

    plt.tight_layout()
    path = out_dir / "decision_quality.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_nc_gain_comparison(data: dict, policies: list[str], out_dir: Path):
    has_new = isinstance(data.get(policies[0], {}).get("mean_nc_gain"), dict)
    if not has_new:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(policies))
    width = 0.35

    chosen_means = [_mean_err(data, p, "mean_nc_gain")[0] for p in policies]
    chosen_errs  = [_mean_err(data, p, "mean_nc_gain")[1] for p in policies]
    greedy_means = [_mean_err(data, p, "mean_greedy_nc_gain")[0] for p in policies]
    greedy_errs  = [_mean_err(data, p, "mean_greedy_nc_gain")[1] for p in policies]

    ax.bar(x - width/2, chosen_means, width, yerr=chosen_errs, capsize=4,
           label="Chosen action", color=[POLICY_COLORS.get(p, "#607D8B") for p in policies],
           alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, greedy_means, width, yerr=greedy_errs, capsize=4,
           label="Greedy oracle", color="lightgray", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=9)
    ax.set_title("NC Gain: Chosen vs Greedy Oracle", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean ΔNC", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = out_dir / "nc_gain_vs_greedy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_anc_comparison(data: dict, policies: list[str], out_dir: Path):
    has_new = isinstance(data.get(policies[0], {}).get("anc_fixed"), dict)
    if not has_new:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(policies))
    width = 0.25

    for i, (key, label, hatch) in enumerate([
        ("final_nc",     "Final NC (snapshot)", ""),
        ("anc_fixed",    "ANC Fixed Horizon",   "//"),
        ("anc_adaptive", "ANC Adaptive Horizon", ".."),
    ]):
        means = [_mean_err(data, p, key)[0] for p in policies]
        errs  = [_mean_err(data, p, key)[1] for p in policies]
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3,
               label=label, hatch=hatch, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=9)
    ax.set_title("NC vs ANC Metrics per Policy", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = out_dir / "anc_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument(
        "--summary", type=Path,
        default=ROOT / "experiments" / "eval_out" / "evaluation_summary.json",
        help="Path to evaluation_summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save plots (defaults to same dir as summary).",
    )
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Summary not found: {args.summary}")
        sys.exit(1)

    data = load_summary(args.summary)
    policies = get_policies(data)
    out_dir = args.output_dir or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Policies found: {policies}")

    plot_main_metrics(data, policies, out_dir)
    plot_decision_quality(data, policies, out_dir)
    plot_nc_gain_comparison(data, policies, out_dir)
    plot_anc_comparison(data, policies, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
