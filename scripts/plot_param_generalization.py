"""Generate evaluation plots and tables from param generalization results.

Produces figures for the evaluation draft (evaluation_draft.tex):

  1. heuristic_ordering.png     -- bar chart: ANC-fix per policy, faceted by pfail
  2. regime_heatmaps.png        -- ANC-fix heatmap (pfail x budget) per policy
  3. topology_comparison.png    -- grouped bar: BA vs ER (vs WS from topo ablation)
  4. anc_vs_budget.png          -- ANC-fix vs budget per policy, one panel per pfail
  5. topology_anc_profiles.png  -- ANC-fix vs budget per topology at mid regime
  6. summary_table.csv          -- machine-readable table for LaTeX

Usage
-----
    python scripts/plot_param_generalization.py
    python scripts/plot_param_generalization.py --out-dir experiments/eval_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

POLICIES = ["greedy", "degree", "betweenness", "risk", "random"]
POLICY_LABELS = {
    "greedy": "Greedy",
    "degree": "Degree",
    "betweenness": "Betweenness",
    "risk": "Risk",
    "random": "Random",
}
COLORS = {
    "greedy":      "#2166ac",
    "degree":      "#4dac26",
    "betweenness": "#7fbc41",
    "risk":        "#d6604d",
    "random":      "#bababa",
}
MARKERS = {
    "greedy": "o", "degree": "s", "betweenness": "^", "risk": "D", "random": "x",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_cells(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    for cell in data["cells"]:
        for pol in POLICIES:
            if pol not in cell["summaries"]:
                continue
            s = cell["summaries"][pol]
            rows.append({
                "alpha":  cell["alpha"],
                "pfail":  cell["pfail"],
                "budget": cell["budget"],
                "policy": pol,
                "anc_fix":  s["anc_fixed_mean"],
                "anc_fix_se": s["anc_fixed_stderr"],
                "anc_adp":  s["anc_adaptive_mean"],
                "final_nc": s["final_nc_mean"],
                "solved":   s["solved_fraction_mean"],
                "rounds":   s.get("rounds_when_solved_mean"),
                "nc_gain":  s["mean_nc_gain_mean"],
                "act_rank": s["mean_action_rank_mean"],
            })
    return pd.DataFrame(rows)


def load_topo_ablation(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    for topo, tdata in data["topologies"].items():
        for pol in POLICIES:
            if pol not in tdata["summaries"]:
                continue
            s = tdata["summaries"][pol]
            rows.append({
                "topology": topo.upper(),
                "policy":   pol,
                "anc_fix":  s["anc_fixed_mean"],
                "anc_fix_se": s["anc_fixed_stderr"],
                "anc_adp":  s["anc_adaptive_mean"],
                "solved":   s["solved_fraction_mean"],
                "regime":   data["regime"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1: Heuristic ordering — bar chart faceted by pfail
# ---------------------------------------------------------------------------

def plot_heuristic_ordering(df: pd.DataFrame, out: Path, label: str = "BA") -> None:
    pfails = sorted(df["pfail"].unique())
    fig, axes = plt.subplots(1, len(pfails), figsize=(4 * len(pfails), 4.5), sharey=True)
    if len(pfails) == 1:
        axes = [axes]

    for ax, pf in zip(axes, pfails):
        sub = df[df["pfail"] == pf].groupby("policy")["anc_fix"].agg(["mean", "sem"]).reindex(POLICIES)
        x = np.arange(len(POLICIES))
        bars = ax.bar(x, sub["mean"], yerr=1.96 * sub["sem"],
                      color=[COLORS[p] for p in POLICIES],
                      capsize=4, width=0.6, error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], rotation=30, ha="right", fontsize=9)
        ax.set_title(f"$p_{{\\mathrm{{fail}}}}={pf}$", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("ANC-fixed (mean ± 95% CI)", fontsize=10)
    fig.suptitle(f"Heuristic ordering — {label} 30–50, averaged over $\\alpha$ and $B$", fontsize=11)
    fig.tight_layout()
    path = out / "heuristic_ordering.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Regime heatmaps — ANC-fix over (pfail x budget) per policy
# ---------------------------------------------------------------------------

def plot_regime_heatmaps(df: pd.DataFrame, out: Path, alpha_mid: float = 0.20) -> None:
    sub = df[np.isclose(df["alpha"], alpha_mid)]
    if sub.empty:
        alphas = sorted(df["alpha"].unique())
        alpha_mid = alphas[len(alphas) // 2]
        sub = df[np.isclose(df["alpha"], alpha_mid)]

    pfails  = sorted(sub["pfail"].unique())
    budgets = sorted(sub["budget"].unique())

    fig, axes = plt.subplots(1, len(POLICIES), figsize=(3.2 * len(POLICIES), 3.5))
    vmin = sub["anc_fix"].min()
    vmax = sub["anc_fix"].max()

    for ax, pol in zip(axes, POLICIES):
        grid = np.zeros((len(pfails), len(budgets)))
        psub = sub[sub["policy"] == pol]
        for i, pf in enumerate(pfails):
            for j, bud in enumerate(budgets):
                val = psub[(np.isclose(psub["pfail"], pf)) & (psub["budget"] == bud)]["anc_fix"]
                grid[i, j] = val.values[0] if len(val) else np.nan

        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto", origin="lower")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels(budgets, fontsize=9)
        ax.set_yticks(range(len(pfails)))
        ax.set_yticklabels(pfails, fontsize=9)
        ax.set_title(POLICY_LABELS[pol], fontsize=10)
        ax.set_xlabel("Budget $B$", fontsize=9)
        for i in range(len(pfails)):
            for j in range(len(budgets)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black")

    axes[0].set_ylabel("$p_{\\mathrm{fail}}$", fontsize=10)
    fig.suptitle(f"ANC-fixed over regime grid ($\\alpha={alpha_mid}$)", fontsize=11)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="ANC-fixed")
    path = out / "regime_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Topology comparison — BA vs ER (+ WS from topo ablation)
# ---------------------------------------------------------------------------

def plot_topology_comparison(
    df_ba: pd.DataFrame,
    df_er: pd.DataFrame,
    df_topo: pd.DataFrame,
    out: Path,
) -> None:
    # Average each over the cells that are common to both BA and ER grids
    # (use the BA 3x3x4 grid as reference)
    ba_cells = set(zip(df_ba["alpha"], df_ba["pfail"], df_ba["budget"]))
    er_match = df_er[df_er.apply(
        lambda r: (r["alpha"], r["pfail"], r["budget"]) in ba_cells, axis=1
    )]

    def mean_anc(df):
        return df.groupby("policy")["anc_fix"].agg(["mean", "sem"]).reindex(POLICIES)

    ba_agg = mean_anc(df_ba)
    er_agg = mean_anc(er_match)
    ws_agg = df_topo[df_topo["topology"] == "WS"].set_index("policy")[["anc_fix", "anc_fix_se"]].reindex(POLICIES)
    ws_agg.columns = ["mean", "sem"]

    x = np.arange(len(POLICIES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, (agg, label, offset) in enumerate([
        (ba_agg, "BA 30–50", -w),
        (er_agg, "ER 30–50",  0),
        (ws_agg, "WS 30–50 (single regime)", w),
    ]):
        ax.bar(x + offset, agg["mean"], width=w, yerr=1.96 * agg["sem"],
               label=label, capsize=3, alpha=0.85,
               color=[COLORS[p] for p in POLICIES] if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=10)
    ax.set_ylabel("ANC-fixed (mean ± 95% CI)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Topology comparison: BA vs ER vs WS", fontsize=11)
    fig.tight_layout()
    path = out / "topology_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: ANC-fix vs budget, faceted by pfail — for a single topology
# ---------------------------------------------------------------------------

def plot_anc_vs_budget(df: pd.DataFrame, out: Path, label: str = "BA",
                       alpha_mid: float = 0.20) -> None:
    sub = df[np.isclose(df["alpha"], alpha_mid)]
    if sub.empty:
        alphas = sorted(df["alpha"].unique())
        alpha_mid = alphas[len(alphas) // 2]
        sub = df[np.isclose(df["alpha"], alpha_mid)]

    pfails  = sorted(sub["pfail"].unique())
    budgets = sorted(sub["budget"].unique())

    fig, axes = plt.subplots(1, len(pfails), figsize=(4 * len(pfails), 4), sharey=True)
    if len(pfails) == 1:
        axes = [axes]

    for ax, pf in zip(axes, pfails):
        psub = sub[np.isclose(sub["pfail"], pf)]
        for pol in POLICIES:
            pp = psub[psub["policy"] == pol].sort_values("budget")
            ax.plot(pp["budget"], pp["anc_fix"],
                    marker=MARKERS[pol], color=COLORS[pol],
                    label=POLICY_LABELS[pol], linewidth=1.8, markersize=6)
            ax.fill_between(pp["budget"],
                            pp["anc_fix"] - 1.96 * pp["anc_fix_se"],
                            pp["anc_fix"] + 1.96 * pp["anc_fix_se"],
                            color=COLORS[pol], alpha=0.12)
        ax.set_title(f"$p_{{\\mathrm{{fail}}}}={pf}$", fontsize=10)
        ax.set_xlabel("Budget $B$", fontsize=9)
        ax.set_xticks(budgets)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("ANC-fixed", fontsize=10)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"ANC-fixed vs budget — {label} ($\\alpha={alpha_mid}$)", fontsize=11)
    fig.tight_layout()
    path = out / f"anc_vs_budget_{label.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 5: ANC profiles across topologies (BA / ER / WS) at mid regime
# ---------------------------------------------------------------------------

def plot_topology_anc_profiles(
    df_ba: pd.DataFrame,
    df_er: pd.DataFrame,
    df_topo: pd.DataFrame,
    out: Path,
    alpha_mid: float = 0.20,
    pfail_mid: float = 0.15,
) -> None:
    budgets = sorted(df_ba["budget"].unique())

    def get_anc_by_budget(df, alpha, pfail):
        sub = df[np.isclose(df["alpha"], alpha) & np.isclose(df["pfail"], pfail)]
        return sub

    ba_sub = get_anc_by_budget(df_ba, alpha_mid, pfail_mid)
    er_sub = get_anc_by_budget(df_er, alpha_mid, pfail_mid)

    topo_regime = df_topo["regime"].iloc[0]
    ws_vals = df_topo[df_topo["topology"] == "WS"].set_index("policy")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    titles = [f"BA 30–50", f"ER 30–50", f"WS 30–50\n(single regime: α={topo_regime['alpha']}, p={topo_regime['pfail']}, B={topo_regime['budget']})"]

    for ax, (sub, tname) in zip(axes, [
        (ba_sub, "BA"), (er_sub, "ER"), (None, "WS")
    ]):
        for pol in POLICIES:
            if tname == "WS":
                if pol not in ws_vals.index:
                    continue
                val = ws_vals.loc[pol, "anc_fix"]
                se  = ws_vals.loc[pol, "anc_fix_se"]
                ax.axhline(val, color=COLORS[pol], linestyle="--",
                           label=f"{POLICY_LABELS[pol]} ({val:.3f})", linewidth=1.5)
            else:
                pp = sub[sub["policy"] == pol].sort_values("budget")
                ax.plot(pp["budget"], pp["anc_fix"],
                        marker=MARKERS[pol], color=COLORS[pol],
                        label=POLICY_LABELS[pol], linewidth=1.8, markersize=6)
                ax.fill_between(pp["budget"],
                                pp["anc_fix"] - 1.96 * pp["anc_fix_se"],
                                pp["anc_fix"] + 1.96 * pp["anc_fix_se"],
                                color=COLORS[pol], alpha=0.12)
        ax.set_xticks(budgets)
        ax.set_xlabel("Budget $B$" if tname != "WS" else "")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("ANC-fixed", fontsize=10)
    fig.suptitle(f"ANC-fixed vs budget by topology ($\\alpha={alpha_mid}$, $p_{{\\mathrm{{fail}}}}={pfail_mid}$)", fontsize=11)
    fig.tight_layout()
    path = out / "topology_anc_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Table: summary CSV
# ---------------------------------------------------------------------------

def write_summary_table(df_ba: pd.DataFrame, df_er: pd.DataFrame,
                        df_topo: pd.DataFrame, out: Path) -> None:
    rows = []
    for pol in POLICIES:
        row = {"policy": POLICY_LABELS[pol]}
        for setting, df in [("BA_30_50", df_ba), ("ER_30_50", df_er)]:
            sub = df[df["policy"] == pol]
            row[f"{setting}_anc_fix_mean"] = round(sub["anc_fix"].mean(), 4)
            row[f"{setting}_anc_fix_se"]   = round(sub["anc_fix_se"].mean(), 4)
            row[f"{setting}_solved_mean"]  = round(sub["solved"].mean(), 4)
        ws = df_topo[(df_topo["topology"] == "WS") & (df_topo["policy"] == pol)]
        if not ws.empty:
            row["WS_topo_anc_fix"] = round(ws["anc_fix"].values[0], 4)
            row["WS_topo_solved"]  = round(ws["solved"].values[0], 4)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    path = out / "summary_table.csv"
    df_out.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(df_out.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot param generalization results.")
    p.add_argument("--ba-json", type=Path,
                   default=ROOT / "experiments" / "eval_param_generalization" / "ba_30_50" / "param_generalization_summary.json")
    p.add_argument("--er-json", type=Path,
                   default=ROOT / "experiments" / "eval_param_generalization" / "er_30_50" / "param_generalization_summary.json")
    p.add_argument("--topo-json", type=Path,
                   default=ROOT / "experiments" / "eval_topology_ablation" / "topology_ablation_summary.json")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "experiments" / "eval_plots" / "param_generalization")
    p.add_argument("--alpha-mid", type=float, default=0.20)
    p.add_argument("--pfail-mid", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading BA:   {args.ba_json}")
    df_ba = load_cells(args.ba_json)
    print(f"  {len(df_ba)} rows, {df_ba['alpha'].nunique()} alpha x "
          f"{df_ba['pfail'].nunique()} pfail x {df_ba['budget'].nunique()} budget")

    print(f"Loading ER:   {args.er_json}")
    df_er = load_cells(args.er_json)
    print(f"  {len(df_er)} rows, {df_er['alpha'].nunique()} alpha x "
          f"{df_er['pfail'].nunique()} pfail x {df_er['budget'].nunique()} budget")

    print(f"Loading topo: {args.topo_json}")
    df_topo = load_topo_ablation(args.topo_json)
    print(f"  Topologies: {df_topo['topology'].unique()}")

    print(f"\nOutputting to: {args.out_dir}\n")

    print("1. Heuristic ordering (BA)...")
    plot_heuristic_ordering(df_ba, args.out_dir, label="BA")

    print("2. Regime heatmaps (BA)...")
    plot_regime_heatmaps(df_ba, args.out_dir, alpha_mid=args.alpha_mid)

    print("3. Topology comparison (BA vs ER vs WS)...")
    plot_topology_comparison(df_ba, df_er, df_topo, args.out_dir)

    print("4. ANC vs budget (BA)...")
    plot_anc_vs_budget(df_ba, args.out_dir, label="BA", alpha_mid=args.alpha_mid)

    print("4b. ANC vs budget (ER)...")
    plot_anc_vs_budget(df_er, args.out_dir, label="ER", alpha_mid=args.alpha_mid)

    print("5. Topology ANC profiles (BA / ER / WS)...")
    plot_topology_anc_profiles(df_ba, df_er, df_topo, args.out_dir,
                               alpha_mid=args.alpha_mid, pfail_mid=args.pfail_mid)

    print("6. Summary table...")
    write_summary_table(df_ba, df_er, df_topo, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
