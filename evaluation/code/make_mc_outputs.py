"""Figure: optimization minus baseline (adjusted input) across all evaluation paths.

Usage: python make_mc_outputs.py <run900 dir> <mc_all dir> <out dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

run, mcd, out = map(Path, sys.argv[1:4])
res = pd.read_csv(run / "evaluation_all.csv")
L = res.pivot(index="path", columns="policy", values="loss")
labels, vals, err = [], [], []
for p, lab in [("scenarios_expected", "Planning\nscenarios"), ("holdout", "Holdout\nweek"),
               ("holdout_plus25", "Holdout week\n(+25% lost)")]:
    labels.append(lab)
    vals.append((L.loc[p, "M-rec"] - L.loc[p, "B-rec"]) / L.loc[p, "B-rec"] * 100)
    err.append(0)
pdiff = pd.read_csv(mcd / "mc_paired_differences.csv").set_index("comparison")
r = pdiff.loc["M-rec minus B-rec"]
base = pd.read_csv(mcd / "mc_summary.csv").set_index("policy").loc["B-rec", "loss"]
labels.append("Monte Carlo\n(1,990 paths)")
vals.append(r.mean_diff / base * 100)
err.append((r.ci_high - r.mean_diff) / base * 100)

BLUE, INK, MUTED, GRID, SURF = "#2a78d6", "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
fig, ax = plt.subplots(figsize=(6.4, 3.3), facecolor=SURF)
ax.set_facecolor(SURF)
x = range(len(vals))
ax.bar(x, vals, width=0.55, color=BLUE, zorder=3)
ax.errorbar(x[-1], vals[-1], yerr=err[-1], color=INK, capsize=4, lw=1.2, zorder=4)
for xi, v in zip(x, vals):
    ax.text(xi, v + (0.25 if v >= 0 else -0.25), (f"{v:+.2f}%" if abs(v) < 0.1 else f"{v:+.1f}%"), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(list(x), labels, fontsize=8.5, color=INK)
ax.set_ylabel("Loss change vs. baseline (%)", fontsize=9, color=INK)
ax.set_ylim(-4.6, 2.2)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0, labelsize=8, colors=MUTED)
ax.text(0.01, 0.03, "Below zero: optimization better", transform=ax.transAxes, fontsize=8, color=MUTED)
ax.set_title("Optimization vs. rule-based baseline (stockout-adjusted input)",
             fontsize=10, color=INK, loc="left")
fig.tight_layout()
fig.savefig(out / "fig_milp_vs_baseline.png", dpi=220, facecolor=SURF)
fig.savefig(out / "fig_milp_vs_baseline.pdf", facecolor=SURF)
print([round(v, 2) for v in vals], round(err[-1], 2))
