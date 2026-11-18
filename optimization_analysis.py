"""Group 21L: integrate the recorded M4 plan and M5 sensitivity experiments.

Default execution audits existing results and creates report figures/tables.
Optional solve flags use the accompanying original Gurobi implementation.
"""
from pathlib import Path
import argparse
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, TEAL, ORANGE, GREY = "#234B73", "#138578", "#D47A34", "#929CA5"
ORDER = ["low", "typical", "high"]
BASE_WEIGHTS = dict(alpha=5, beta=1, h=.2, tau=.5)


def save(fig, name):
    fig.savefig(ROOT / "figures" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(ROOT / "figures" / (name + ".png"), dpi=220, bbox_inches="tight")
    latex_figures = ROOT / "latex/figures"
    latex_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(latex_figures / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def audit_results():
    """Reconcile loss and physical balances using recorded full-precision CSVs."""
    base = ROOT / "results/base_case"
    original = pd.read_csv(base / "scenario_summary.csv").set_index("scenario").loc[ORDER]
    sensroot = ROOT / "results/sensitivity/runs"
    cases = pd.read_csv(ROOT / "results/sensitivity/tables/sensitivity_summary_by_case.csv")
    rows = []
    for folder in [base] + sorted(sensroot.iterdir()):
        if not folder.is_dir():
            continue
        sc = pd.read_csv(folder / "scenario_summary.csv")
        u, w, l, terminal = (sc[c] for c in ["unmet_demand", "expired_stock", "overnight_carryover", "day7_new_stock"])
        config = json.loads((folder / ("solver_summary.json" if folder == base else "run_summary.json")).read_text())
        weights = BASE_WEIGHTS if folder == base else config["model_settings"]
        loss = weights["alpha"] * u + weights["beta"] * w + weights["h"] * l + weights["tau"] * terminal
        errors = [np.max(np.abs(sc.sales + u - sc.demand)), np.max(np.abs(loss - sc.loss))]
        target = config["objective_weighted_loss"] if folder == base else config["objective"]
        errors.append(abs(float((sc.probability * loss).sum()) - target))
        if max(errors) > 1e-5:
            raise ValueError(f"Recorded result reconciliation failed: {folder}")
        rows.append(dict(case=folder.name, max_reconciliation_error=max(errors)))
    pd.DataFrame(rows).to_csv(ROOT / "results/report_audit.csv", index=False)
    return original, cases.set_index("case_id")


def make_figures(sc, cases):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
    x = np.arange(3)
    ax.bar(x, sc.unselected_product_demand, color=GREY, label="Excluded products")
    ax.bar(x, sc.selected_product_unmet, bottom=sc.unselected_product_demand,
           color=ORANGE, label="Selected products, insufficient stock")
    for i, row in enumerate(sc.itertuples()):
        ax.text(i, row.unmet_demand + 8, f"{row.unmet_demand:.1f}", ha="center")
    ax.set(xticks=x, xticklabels=["Low", "Typical", "High"], ylabel="Unmet demand (normalized units)",
           ylim=(0, 440), title="Why does demand remain unserved?")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig01_shortage_sources")

    cap = pd.read_csv(ROOT / "results/base_case/store_daily_capacity.csv")
    # Use recorded column labels and retain a single low-demand scenario.
    daycol = "dt" if "dt" in cap else "t"
    matrix = cap[cap.scenario == "low"].pivot(index="store_id", columns=daycol, values="utilization")
    regional = pd.read_csv(ROOT / "results/base_case/regional_daily_supply.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), layout="constrained", height_ratios=[1, 1.2])
    axes[0].imshow(matrix.to_numpy(), cmap="Blues", vmin=.65, vmax=1, aspect="auto")
    axes[0].set(yticks=range(3), yticklabels=[str(v) for v in matrix.index],
                xticks=range(7), xticklabels=range(1, 8), ylabel="Store", title="Low-demand carryover uses store space")
    for i in range(3):
        for j in range(7):
            value=matrix.iloc[i,j]
            axes[0].text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value>.85 else "black")
    supplycol = "delivery"
    axes[1].bar(range(1,8), regional[supplycol], color=TEAL, label="New deliveries")
    axes[1].axhline(144.43, color=ORANGE, ls="--", label="Regional limit 144.43")
    axes[1].set(xlabel="Planning day", ylabel="Normalized units", ylim=(0,175))
    axes[1].legend(frameon=False, ncol=2, loc="upper center")
    save(fig, "fig02_inventory_and_supply")

    ids = ["K_15", "base", "K_25", "K_29"]
    d = cases.loc[ids]
    fig, axes = plt.subplots(1,2,figsize=(8,3.2),layout="constrained")
    axes[0].plot([15,20,25,29],100*d.exp_fill_rate,marker="o",color=TEAL,lw=2)
    for k,val in zip([15,20,25,29],100*d.exp_fill_rate):
        axes[0].annotate(f"{val:.1f}%",(k,val),xytext=(0,8),textcoords="offset points",ha="center")
    axes[0].set(xlabel="Assortment limit per store",ylabel="Aggregate fill rate (%)",xticks=[15,20,25,29],ylim=(70,87),title="Gains taper after 25 products")
    axes[1].bar(range(4),d.exp_unselected_demand,color=GREY,label="Excluded products")
    axes[1].bar(range(4),d.exp_selected_unmet,bottom=d.exp_unselected_demand,color=ORANGE,label="Selected products")
    axes[1].set(xticks=range(4),xticklabels=[15,20,25,29],xlabel="Assortment limit per store",ylabel="Expected unmet units",title="Shortages shift within the assortment")
    axes[1].legend(frameon=False,fontsize=9)
    save(fig,"fig03_assortment")

    ids=["K_15","K_25","K_29","capacity_0.85","capacity_1.2","supply_0.85","supply_1.2"]
    labels=["K = 15","K = 25","K = 29","Storage -15%","Storage +20%","Supply -15%","Supply +20%"]
    fig,ax=plt.subplots(figsize=(7.5,3.4),layout="constrained")
    changes=100*(cases.loc[ids].exp_fill_rate-cases.loc["base"].exp_fill_rate)
    ax.barh(labels,changes,color=[TEAL if v>=0 else ORANGE for v in changes])
    for i,v in enumerate(changes):
        ax.text(v+(.15 if v>=0 else -.15),i,f"{v:+.2f}",ha="left" if v>=0 else "right",va="center")
    ax.axvline(0,color=GREY,lw=.8);ax.invert_yaxis()
    ax.set(xlim=(-10,6),xlabel="Change in aggregate fill rate (percentage points)",title="Assortment and storage drive the modeled service level")
    save(fig,"fig04_resource_effects")

    fig,ax=plt.subplots(figsize=(7.5,3.3),layout="constrained")
    ids=["base","K_25","K_29","capacity_1.2","supply_1.2"]
    labels=["Base","K = 25","K = 29","Storage +20%","Supply +20%"]
    offsets=[(-45,12),(12,-14),(12,15),(-85,8),(-60,-17)]
    for key,label,offset in zip(ids,labels,offsets):
        r=cases.loc[key];ax.scatter(r.exp_expired,100*r.exp_fill_rate,s=65,color=TEAL if key!="base" else BLUE)
        ax.annotate(label,(r.exp_expired,100*r.exp_fill_rate),xytext=offset,textcoords="offset points",fontsize=10)
    ax.set(xlabel="Expected expired stock (normalized units)",ylabel="Aggregate fill rate (%)",xlim=(-.3,10.5),ylim=(78,86),title="Storage expansion improves service and increases expiration")
    save(fig,"fig05_service_waste")

    ids=["alpha_2","alpha_10","h_0","h_0.5","tau_0.25","tau_1"]
    table=cases.loc[ids,["base_weight_loss","exp_fill_rate","exp_expired","mip_gap","assortment_changes_vs_base"]].copy()
    table.to_csv(ROOT / "results/penalty_comparison.csv")
    cases.loc[["base","K_15","K_25","K_29","capacity_0.85","capacity_1.2","supply_0.85","supply_1.2"]].to_csv(ROOT / "results/resource_comparison.csv")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-base",action="store_true")
    parser.add_argument("--solve-sensitivity",action="store_true")
    parser.add_argument("--time-limit",type=float,default=60)
    args=parser.parse_args()
    if args.solve_base or args.solve_sensitivity:
        import gurobi_model as gm
        import sensitivity_analysis as sa
        if args.solve_base:
            gm.solve_case(gm.load_and_audit_inputs(ROOT/"data",sa.STORES,sa.PRODUCTS,sa.DATES),
                          sa.BASE_SETTINGS,sa.SOLVER_SETTINGS,ROOT/"results/base_case")
        if args.solve_sensitivity:
            sa.run_all(sa.DEFAULT_GROUPS,args.time_limit,force=True)
            sa.analyze()
    (ROOT/"figures").mkdir(exist_ok=True)
    sc,cases=audit_results()
    make_figures(sc,cases)
    print("Report figures and verified summary tables are available in figures/ and results/.")


if __name__ == "__main__":
    main()
