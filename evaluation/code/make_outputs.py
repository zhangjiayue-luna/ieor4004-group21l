"""Make the Milestone 5 comparison tables and figure from run_evaluation.py output.

Usage: python make_outputs.py <run output dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

out = Path(sys.argv[1])
res = pd.read_csv(out / "evaluation_all.csv")
PATHS = {"scenarios_expected": "Planning scenarios\n(expected)",
         "holdout": "Holdout week",
         "holdout_plus25": "Holdout week\n(+25% lost demand)"}
POL = ["B-raw", "M-raw", "B-rec", "M-rec", "M-rec-Gurobi"]
res = res[res.path.isin(PATHS)]

# ---- table 1: 2x2 on every path
cols = ["fill_rate", "unmet", "unmet_unselected", "expired", "carryover", "day7_stock",
        "loss", "capacity_overflow"]
t1 = res.set_index(["path", "policy"])[cols].reindex(
    pd.MultiIndex.from_product([list(PATHS), POL])).reset_index()
t1.columns = ["path", "policy"] + cols
t1.to_csv(out / "table_2x2_all_paths.csv", index=False)

# ---- table 2: decomposition of loss differences
rows = []
for p in PATHS:
    L = res[res.path == p].set_index("policy").loss
    rows.append(dict(path=p,
                     recovery_effect_baseline=L["B-rec"] - L["B-raw"],
                     recovery_effect_milp=L["M-rec"] - L["M-raw"],
                     optimization_effect_raw=L["M-raw"] - L["B-raw"],
                     optimization_effect_rec=L["M-rec"] - L["B-rec"],
                     optimization_effect_rec_pct=(L["M-rec"] - L["B-rec"]) / L["B-rec"] * 100,
                     gurobi_vs_baseline_rec=L["M-rec-Gurobi"] - L["B-rec"],
                     gurobi_vs_baseline_rec_pct=(L["M-rec-Gurobi"] - L["B-rec"]) / L["B-rec"] * 100))
pd.DataFrame(rows).to_csv(out / "table_effects.csv", index=False)

# ---- figure: weighted loss, 2x2, three evaluation paths
BLUE, ORANGE = "#2a78d6", "#eb6834"      # baseline, optimization (validated pair)
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True, facecolor=SURF)
for ax, (tag, title) in zip(axes, [("raw", "Input: raw sales"),
                                   ("rec", "Input: stockout-adjusted demand")]):
    ax.set_facecolor(SURF)
    x = range(len(PATHS))
    w = 0.36
    for k, (pol, col, lab) in enumerate([(f"B-{tag}", BLUE, "Rule-based baseline"),
                                         (f"M-{tag}", ORANGE, "Optimization (MILP)")]):
        vals = [res[(res.path == p) & (res.policy == pol)].loss.item() for p in PATHS]
        xs = [i + (k - 0.5) * (w + 0.02) for i in x]
        ax.bar(xs, vals, width=w, color=col, label=lab, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 15, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(x), list(PATHS.values()), fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Weighted loss J (lower is better)", fontsize=9, color=INK)
axes[0].set_ylim(0, 1950)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.99, 1.0))
fig.suptitle("All plans scored on the same stockout-adjusted demand", x=0.01, ha="left",
             fontsize=10.5, color=INK, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out / "fig_loss_2x2.png", dpi=220, facecolor=SURF)
fig.savefig(out / "fig_loss_2x2.pdf", facecolor=SURF)
print(pd.DataFrame(rows).round(2).to_string(index=False))
