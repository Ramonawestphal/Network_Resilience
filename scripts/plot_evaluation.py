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
        "final_nc": ("final_nc_mean", "final_nc_stderr"),
        "solved_fraction": ("solved_fraction_mean", None),
        "rounds": ("rounds_mean", None),
    }
    if key in flat_map:
        mk, ek = flat_map[key]
        if mk not in p:
            return None, None
        mean = float(p[mk]) if p[mk] is not None else None
        err = float(p[ek]) if ek and p.get(ek) is not None else None
        return mean, err
    return None, None


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
    metrics = [
        ("final_nc",        "Final NC (snapshot)",         "NC",       "final_nc"),
        ("anc_fixed",       "ANC Fixed Horizon",           "ANC",      "anc_fixed"),
        ("anc_adaptive",    "ANC Adaptive Horizon",        "ANC",      "anc_adaptive"),
        ("solved_fraction", "Solved Fraction",             "Fraction", "solved_fraction"),
        ("rounds",          "Mean Rounds Taken",           "Rounds",   "rounds"),
    ]

    for key, title, ylabel, filename in metrics:
        metric_pairs = [_mean_err(data, p, key) for p in policies]
        if any(mean is None or err is None for mean, err in metric_pairs):
            print(f"Skipping {filename}: missing metric data in summary.")
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = [mean for mean, _ in metric_pairs]
        errs = [err for _, err in metric_pairs]
        bar_chart(ax, policies, vals, errs, title, ylabel, POLICY_COLORS)
        plt.tight_layout()
        path = out_dir / f"{filename}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # b_star — separate file
    fig, ax = plt.subplots(figsize=(7, 4))
    b_stars = [_scalar(data, p, "b_star") for p in policies]
    vals = [b if b is not None else 0 for b in b_stars]
    labels = [str(int(b)) if b is not None else "N/A" for b in b_stars]
    bars = bar_chart(ax, policies, vals, [0] * len(policies),
                     "Minimum Budget (b*)", "Budget", POLICY_COLORS)
    for bar, label in zip(bars, labels):
        if label == "N/A":
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "N/A",
                    ha="center", va="bottom", fontsize=8, color="gray")
    plt.tight_layout()
    path = out_dir / "b_star.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_decision_quality(data: dict, policies: list[str], out_dir: Path):
    new_metrics = [
        ("mean_degree_ratio",   "Degree Ratio\n(chosen / max failed degree)",    "Ratio", "degree_ratio"),
        ("mean_overload_risk",  "Overload Risk\n(max load/capacity of neighbors)","Risk",  "overload_risk"),
        ("mean_nc_gain",        "NC Gain of Chosen Action",                       "ΔNC",   "nc_gain"),
        ("mean_greedy_nc_gain", "NC Gain of Greedy Oracle",                       "ΔNC",   "greedy_nc_gain"),
        ("mean_action_rank",    "Action Rank\n(1 = optimal)",                     "Rank",  "action_rank"),
    ]

    has_new = any(
        isinstance(data.get(policies[0], {}).get(key), dict)
        for key, _, _, _ in new_metrics
    )
    if not has_new:
        print("Decision-quality metrics not in summary (re-run evaluation to generate them).")
        return

    for key, title, ylabel, filename in new_metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        metric_pairs = [_mean_err(data, p, key) for p in policies]
        if any(mean is None or err is None for mean, err in metric_pairs):
            print(f"Skipping {filename}: missing metric data in summary.")
            plt.close()
            continue
        vals = [mean for mean, _ in metric_pairs]
        errs = [err for _, err in metric_pairs]
        bar_chart(ax, policies, vals, errs, title, ylabel, POLICY_COLORS)
        plt.tight_layout()
        path = out_dir / f"{filename}.png"
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
    plotted_any = False

    for i, (key, label, hatch) in enumerate([
        ("final_nc",     "Final NC (snapshot)", ""),
        ("anc_fixed",    "ANC Fixed Horizon",   "//"),
        ("anc_adaptive", "ANC Adaptive Horizon", ".."),
    ]):
        metric_pairs = [_mean_err(data, p, key) for p in policies]
        if any(mean is None or err is None for mean, err in metric_pairs):
            continue
        means = [mean for mean, _ in metric_pairs]
        errs  = [err for _, err in metric_pairs]
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3,
               label=label, hatch=hatch, alpha=0.85, edgecolor="white")
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

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
    if not policies:
        sys.stderr.write(
            f"No plottable policies found in summary: {args.summary}\n"
        )
        sys.exit(1)
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
