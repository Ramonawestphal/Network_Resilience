"""Plot Tier 1 (topology ablation) and Tier 2 (OOD) evaluation results.

Produces:
  experiments/eval_plots/topology_ablation.png  — BA vs ER vs WS, ANC-fixed per policy
  experiments/eval_plots/ood_ieee300.png        — IEEE 300-bus, ANC-fixed per policy
  experiments/eval_plots/combined_tiers.png     — both tiers side-by-side (paper figure)

Usage
-----
    python scripts/plot_evaluation_tiers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_DEFAULT_TOPO_JSON = ROOT / "experiments" / "eval_topology_ablation" / "topology_ablation_summary.json"
_DEFAULT_OOD_JSON  = ROOT / "experiments" / "eval_real_world" / "ieee300" / "evaluation_summary.json"
OUT_DIR            = ROOT / "experiments" / "eval_plots"

POLICY_ORDER  = ["rl", "greedy", "betweenness", "degree", "risk", "random"]
POLICY_LABELS = {
    "rl":          "RL",
    "greedy":      "Greedy",
    "betweenness": "Betweenness",
    "degree":      "Degree",
    "risk":        "Risk",
    "random":      "Random",
}
POLICY_COLORS = {
    "rl":          "#2196F3",
    "greedy":      "#4CAF50",
    "betweenness": "#9C27B0",
    "degree":      "#FF9800",
    "risk":        "#F44336",
    "random":      "#9E9E9E",
}
TOPOLOGY_LABELS = {"ba": "BA", "er": "ER", "ws": "WS"}
TOPOLOGY_HATCHES = {"ba": "", "er": "//", "ws": ".."}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _anc(summaries: dict, policy: str) -> tuple[float, float]:
    s = summaries.get(policy, {})
    return float(s.get("anc_fixed_mean", 0.0)), float(s.get("anc_fixed_stderr", 0.0))


def _spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Tier 1: topology ablation — grouped bar chart (policies × topologies)
# ---------------------------------------------------------------------------

def plot_topology_ablation(data: dict, out_dir: Path) -> Path:
    topologies = [t for t in ["ba", "er", "ws"] if t in data["topologies"]]
    policies   = [p for p in POLICY_ORDER if p in data["topologies"][topologies[0]]["summaries"]]

    n_policies = len(policies)
    n_topos    = len(topologies)
    x          = np.arange(n_policies)
    total_width = 0.7
    bar_width   = total_width / n_topos
    offsets     = np.linspace(-total_width / 2 + bar_width / 2,
                               total_width / 2 - bar_width / 2, n_topos)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for topo, offset in zip(topologies, offsets):
        summaries = data["topologies"][topo]["summaries"]
        means = [_anc(summaries, p)[0] for p in policies]
        errs  = [_anc(summaries, p)[1] for p in policies]
        ax.bar(
            x + offset, means, bar_width,
            yerr=errs, capsize=3,
            label=TOPOLOGY_LABELS[topo],
            hatch=TOPOLOGY_HATCHES[topo],
            alpha=0.88, edgecolor="white", linewidth=0.5,
            color=[POLICY_COLORS[p] for p in policies],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    ax.set_title("Topology Ablation: BA vs ER vs WS  [n ∈ [30, 50], avg degree 4]",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Topology", fontsize=9, title_fontsize=9)
    _spine_clean(ax)
    plt.tight_layout()

    path = out_dir / "topology_ablation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


def _plot_topology_ablation_tagged(data: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    topologies = [t for t in ["ba", "er", "ws"] if t in data["topologies"]]
    policies   = [p for p in POLICY_ORDER if p in data["topologies"][topologies[0]]["summaries"]]
    n_policies = len(policies)
    n_topos    = len(topologies)
    x          = np.arange(n_policies)
    total_width = 0.7
    bar_width   = total_width / n_topos
    offsets     = np.linspace(-total_width / 2 + bar_width / 2,
                               total_width / 2 - bar_width / 2, n_topos)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for topo, offset in zip(topologies, offsets):
        summaries = data["topologies"][topo]["summaries"]
        means = [_anc(summaries, p)[0] for p in policies]
        errs  = [_anc(summaries, p)[1] for p in policies]
        ax.bar(x + offset, means, bar_width, yerr=errs, capsize=3,
               label=TOPOLOGY_LABELS[topo], hatch=TOPOLOGY_HATCHES[topo],
               alpha=0.88, edgecolor="white", linewidth=0.5,
               color=[POLICY_COLORS[p] for p in policies])
    r = data.get("regime", {})
    subtitle = (f"α={r.get('alpha')}  p_fail={r.get('pfail')}  B={r.get('budget')}"
                if r else "")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    ax.set_title(f"Topology Ablation: BA vs ER vs WS  [n ∈ [30, 50]]\n{subtitle}",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Topology", fontsize=9, title_fontsize=9)
    _spine_clean(ax)
    plt.tight_layout()
    path = out_dir / f"topology_ablation{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Tier 2: OOD — single bar chart (policies on IEEE 300-bus)
# ---------------------------------------------------------------------------

def plot_ood(data: dict, out_dir: Path) -> Path:
    summaries = data["summaries"]
    policies  = [p for p in POLICY_ORDER if p in summaries]

    means = [_anc(summaries, p)[0] for p in policies]
    errs  = [_anc(summaries, p)[1] for p in policies]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(policies))
    ax.bar(
        x, means, 0.6,
        yerr=errs, capsize=4,
        color=[POLICY_COLORS[p] for p in policies],
        alpha=0.88, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    n = data["graph"]["num_nodes"]
    m = data["graph"]["num_edges"]
    ax.set_title(f"OOD Evaluation: IEEE 300-bus  [n={n}, m={m}]",
                 fontsize=11, fontweight="bold")
    _spine_clean(ax)
    plt.tight_layout()

    path = out_dir / "ood_ieee300.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Combined: both tiers in one figure (paper-ready)
# ---------------------------------------------------------------------------

def plot_combined(ablation_data: dict, ood_data: dict, out_dir: Path) -> Path:
    topologies = [t for t in ["ba", "er", "ws"] if t in ablation_data["topologies"]]
    policies   = [p for p in POLICY_ORDER
                  if p in ablation_data["topologies"][topologies[0]]["summaries"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                              gridspec_kw={"width_ratios": [2, 1]})

    # --- Left: topology ablation ---
    ax = axes[0]
    n_topos    = len(topologies)
    x          = np.arange(len(policies))
    total_width = 0.7
    bar_width   = total_width / n_topos
    offsets     = np.linspace(-total_width / 2 + bar_width / 2,
                               total_width / 2 - bar_width / 2, n_topos)

    handles = []
    for topo, offset in zip(topologies, offsets):
        summaries = ablation_data["topologies"][topo]["summaries"]
        means = [_anc(summaries, p)[0] for p in policies]
        errs  = [_anc(summaries, p)[1] for p in policies]
        bars = ax.bar(
            x + offset, means, bar_width,
            yerr=errs, capsize=3,
            label=TOPOLOGY_LABELS[topo],
            hatch=TOPOLOGY_HATCHES[topo],
            alpha=0.88, edgecolor="white", linewidth=0.5,
            color=[POLICY_COLORS[p] for p in policies],
        )
        handles.append(bars)

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    ax.set_title("(a) Topology Ablation  [n ∈ [30,50]]", fontsize=11, fontweight="bold")
    ax.legend(title="Topology", fontsize=9, title_fontsize=9)
    _spine_clean(ax)

    # --- Right: OOD ---
    ax = axes[1]
    ood_summaries = ood_data["summaries"]
    ood_policies  = [p for p in POLICY_ORDER if p in ood_summaries]
    means = [_anc(ood_summaries, p)[0] for p in ood_policies]
    errs  = [_anc(ood_summaries, p)[1] for p in ood_policies]
    x2 = np.arange(len(ood_policies))
    ax.bar(
        x2, means, 0.6,
        yerr=errs, capsize=4,
        color=[POLICY_COLORS[p] for p in ood_policies],
        alpha=0.88, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(x2)
    ax.set_xticklabels([POLICY_LABELS[p] for p in ood_policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    n = ood_data["graph"]["num_nodes"]
    ax.set_title(f"(b) OOD: IEEE 300-bus  [n={n}]", fontsize=11, fontweight="bold")
    _spine_clean(ax)

    plt.tight_layout()
    path = out_dir / "combined_tiers.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


def _plot_ood_tagged(data: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = data["summaries"]
    policies  = [p for p in POLICY_ORDER if p in summaries]
    means = [_anc(summaries, p)[0] for p in policies]
    errs  = [_anc(summaries, p)[1] for p in policies]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(policies))
    ax.bar(x, means, 0.6, yerr=errs, capsize=4,
           color=[POLICY_COLORS[p] for p in policies],
           alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    n = data["graph"]["num_nodes"]
    m = data["graph"]["num_edges"]
    r = data.get("regime", {})
    subtitle = (f"α={r.get('alpha')}  p_fail={r.get('pfail')}  B_ref={r.get('budget_ref')}"
                if r else "")
    ax.set_title(f"OOD Evaluation: IEEE 300-bus  [n={n}, m={m}]\n{subtitle}",
                 fontsize=11, fontweight="bold")
    _spine_clean(ax)
    plt.tight_layout()
    path = out_dir / f"ood_ieee300{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


def _plot_combined_tagged(ablation_data: dict, ood_data: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    topologies = [t for t in ["ba", "er", "ws"] if t in ablation_data["topologies"]]
    policies   = [p for p in POLICY_ORDER
                  if p in ablation_data["topologies"][topologies[0]]["summaries"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                              gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    x = np.arange(len(policies))
    total_width = 0.7
    bar_width   = total_width / len(topologies)
    offsets     = np.linspace(-total_width / 2 + bar_width / 2,
                               total_width / 2 - bar_width / 2, len(topologies))
    for topo, offset in zip(topologies, offsets):
        summaries = ablation_data["topologies"][topo]["summaries"]
        means = [_anc(summaries, p)[0] for p in policies]
        errs  = [_anc(summaries, p)[1] for p in policies]
        ax.bar(x + offset, means, bar_width, yerr=errs, capsize=3,
               label=TOPOLOGY_LABELS[topo], hatch=TOPOLOGY_HATCHES[topo],
               alpha=0.88, edgecolor="white", linewidth=0.5,
               color=[POLICY_COLORS[p] for p in policies])
    r = ablation_data.get("regime", {})
    subtitle = (f"α={r.get('alpha')}  p_fail={r.get('pfail')}  B={r.get('budget')}" if r else "")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    ax.set_title(f"(a) Topology Ablation  [n ∈ [30,50]]\n{subtitle}", fontsize=11, fontweight="bold")
    ax.legend(title="Topology", fontsize=9, title_fontsize=9)
    _spine_clean(ax)
    ax = axes[1]
    ood_summaries = ood_data["summaries"]
    ood_policies  = [p for p in POLICY_ORDER if p in ood_summaries]
    means = [_anc(ood_summaries, p)[0] for p in ood_policies]
    errs  = [_anc(ood_summaries, p)[1] for p in ood_policies]
    x2 = np.arange(len(ood_policies))
    ax.bar(x2, means, 0.6, yerr=errs, capsize=4,
           color=[POLICY_COLORS[p] for p in ood_policies],
           alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x2)
    ax.set_xticklabels([POLICY_LABELS[p] for p in ood_policies], fontsize=10)
    ax.set_ylabel("ANC (fixed horizon)", fontsize=10)
    n = ood_data["graph"]["num_nodes"]
    r2 = ood_data.get("regime", {})
    sub2 = (f"α={r2.get('alpha')}  p_fail={r2.get('pfail')}  B_ref={r2.get('budget_ref')}" if r2 else "")
    ax.set_title(f"(b) OOD: IEEE 300-bus  [n={n}]\n{sub2}", fontsize=11, fontweight="bold")
    _spine_clean(ax)
    plt.tight_layout()
    path = out_dir / f"combined_tiers{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot topology ablation and OOD evaluation results.")
    p.add_argument("--topo-json", type=Path, default=_DEFAULT_TOPO_JSON,
                   help="Path to topology_ablation_summary.json.")
    p.add_argument("--ood-json", type=Path, default=_DEFAULT_OOD_JSON,
                   help="Path to OOD evaluation_summary.json.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Output directory for plots.")
    p.add_argument("--tag", type=str, default="",
                   help="Optional tag appended to filenames, e.g. 'a0.25_p0.2_b2'.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print config being plotted
    print(f"Topo JSON : {args.topo_json}")
    print(f"OOD JSON  : {args.ood_json}")
    print(f"Output    : {out_dir}/")

    missing = []
    if not args.topo_json.exists():
        missing.append(str(args.topo_json))
    if not args.ood_json.exists():
        missing.append(str(args.ood_json))
    if missing:
        print("Missing evaluation files — run these first:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    ablation_data = _load(args.topo_json)
    ood_data      = _load(args.ood_json)

    # Print regime info if present in the data
    if "regime" in ablation_data:
        r = ablation_data["regime"]
        print(f"Ablation regime: alpha={r.get('alpha')}  pfail={r.get('pfail')}  budget={r.get('budget')}")
    if "regime" in ood_data:
        r = ood_data["regime"]
        print(f"OOD regime: alpha={r.get('alpha')}  pfail={r.get('pfail')}  budget_ref={r.get('budget_ref')}")

    tag = f"_{args.tag}" if args.tag else ""
    _plot_topology_ablation_tagged(ablation_data, out_dir, tag)
    _plot_ood_tagged(ood_data, out_dir, tag)
    _plot_combined_tagged(ablation_data, ood_data, out_dir, tag)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
