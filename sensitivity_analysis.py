"""Group 21L, Milestone 5: sensitivity analysis of the Milestone 4 MILP.

Owner: Yue Peng.

The script re-solves the Milestone 4 model (imported unchanged from
``gurobi_model.py``) while changing ONE parameter at a time around the base case:

    K (assortment limit)         15, 20, 25, 29
    store capacity C (x base)    0.85, 1.00, 1.20
    regional supply R (x base)   0.85, 1.00, 1.20
    shortage penalty alpha       2, 5, 10
    overnight carryover cost h   0, 0.2, 0.5
    terminal stock cost tau      0.25, 0.5, 1
    demand level (x forecast)    0.8, 1.0, 1.2      (optional "what-if", group "demand")

What it adds on top of a plain re-solve:

* Warm start. Every case receives the base-case plan, repaired to be feasible
  under the new K / capacity / supply, as a second MIP start. For relaxations
  (K, C or R up) this guarantees the new plan is never worse than the base plan,
  so differences are not artefacts of the 300-second time limit.
* Comparable scoring. Changing alpha, h or tau changes the scale of the
  objective, so each plan is also re-scored with the BASE weights
  (alpha=5, beta=1, h=0.2, tau=0.5) and reported with operational metrics.
* Gap awareness. Each run records its MIP gap; objective changes smaller than
  the gaps involved are flagged as not distinguishable from solver tolerance.
* Shadow prices. For each solved plan, the assortment and min-branch binaries
  are fixed and the resulting LP is solved to obtain duals of the store-capacity
  and regional-supply constraints (local, plan-conditional marginal values).

Usage (from the repository root, with a Gurobi licence):

    python sensitivity_analysis.py                       # all six groups, 60 s per case (~15 min)
    python sensitivity_analysis.py --groups K capacity   # only some groups
    python sensitivity_analysis.py --groups demand       # optional what-if group
    python sensitivity_analysis.py --time-limit 300      # longer limit, as in the base case
    python sensitivity_analysis.py --analyze-only        # rebuild tables/figures

Finished runs are cached under results/sensitivity/runs/<case_id>/ and are
skipped on the next call (use --force to re-solve). Tables go to
results/sensitivity/tables/ and figures to results/sensitivity/figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Settings carried over from Group21L_Milestone4.ipynb (keep these identical).
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
M4_BASE_DIR = PROJECT_DIR / "results" / "base_case"
OUT_DIR = PROJECT_DIR / "results" / "sensitivity"

STORES = [18, 145, 235]
PRODUCTS = [21, 70, 79, 94, 117, 118, 136, 140, 207, 213, 215, 247, 249, 296, 366, 422, 496, 500,
            554, 556, 633, 634, 636, 638, 717, 738, 775, 797, 843]
DATES = pd.date_range("2024-06-26", periods=7).strftime("%Y-%m-%d").tolist()
BASE_SETTINGS = {"K": 20, "capacity_by_store": {18: 51.25, 145: 46.93, 235: 46.41},
                 "regional_supply": 144.43, "capacity_scale": 1.0, "supply_scale": 1.0,
                 "alpha": 5.0, "beta": 1.0, "h": 0.2, "tau": 0.5,
                 "probabilities": None}
SOLVER_SETTINGS = {"TimeLimit": 300, "MIPGap": 0.001, "Seed": 21, "Threads": 4}
# Sensitivity runs use a shorter limit than the 300 s base case: every case is
# warm-started from the base plan, so 60 s is enough to obtain a good incumbent.
# The achieved MIP gap is reported for every case. Override with --time-limit.
SENSITIVITY_TIME_LIMIT = 60
STATES = ("O", "V", "A", "Sold", "Snew", "L", "W", "U")

# group -> (parameter key, tested values, axis label)
EXPERIMENTS = {
    "K": ("K", [15, 20, 25, 29], "Assortment limit K (products per store)"),
    "capacity": ("capacity_scale", [0.85, 1.0, 1.2], "Store capacity (x base)"),
    "supply": ("supply_scale", [0.85, 1.0, 1.2], "Regional supply (x base)"),
    "alpha": ("alpha", [2.0, 5.0, 10.0], "Shortage penalty α"),
    "h": ("h", [0.0, 0.2, 0.5], "Overnight carryover cost h"),
    "tau": ("tau", [0.25, 0.5, 1.0], "Day-7 stock cost τ"),
    "demand": ("demand_scale", [0.8, 1.0, 1.2], "Demand level (x forecast)"),
}
DEFAULT_GROUPS = ["K", "capacity", "supply", "alpha", "h", "tau"]
WEIGHT_GROUPS = {"alpha", "h", "tau"}  # objective scale changes -> compare at base weights
BASE_VALUES = {**{k: v for k, v in BASE_SETTINGS.items() if not isinstance(v, dict)},
               "demand_scale": 1.0}


# --------------------------------------------------------------------------
# Case definitions
# --------------------------------------------------------------------------
def fmt_value(v):
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def build_cases(groups):
    """One base case plus every non-base value of each requested group."""
    cases = [dict(case_id="base", group="base", parameter=None, value=None, overrides={},
                  demand_scale=1.0)]
    for g in groups:
        key, values, _ = EXPERIMENTS[g]
        for value in values:
            if math.isclose(float(value), float(BASE_VALUES[key])):
                continue  # the base value is the shared base case
            overrides = {} if key == "demand_scale" else {key: value}
            cases.append(dict(case_id=f"{g}_{fmt_value(value)}", group=g, parameter=key,
                              value=value, overrides=overrides,
                              demand_scale=float(value) if key == "demand_scale" else 1.0))
    return cases


def scaled_data(data, demand_scale):
    if math.isclose(demand_scale, 1.0):
        return data
    out = dict(data)
    out["d"] = {k: v * demand_scale for k, v in data["d"].items()}
    return out


# --------------------------------------------------------------------------
# Plan simulation, repair and scoring (pure Python, no Gurobi needed)
# --------------------------------------------------------------------------
def simulate_plan(d, q, S, P, T, Omega):
    """FIFO two-period-life simulation of a fixed delivery plan (same rules as the MILP)."""
    out = {n: {} for n in (*STATES, "bA", "bO")}
    for s, p, w in product(S, P, Omega):
        old = 0.0
        for t in T:
            k = (s, p, t, w)
            D, new = d[k], q[s, p, t]
            avail = old + new
            sold_old, sales = min(old, D), min(avail, D)
            sold_new = max(sales - sold_old, 0.0)
            left = max(new - sold_new, 0.0)
            vals = dict(O=old, V=avail, A=sales, Sold=sold_old, Snew=sold_new, L=left,
                        W=max(old - sold_old, 0.0), U=max(D - sales, 0.0),
                        bA=0 if avail <= D else 1, bO=0 if old <= D else 1)
            for n, x in vals.items():
                out[n][k] = x
            old = left
    return out


def plan_objective(sim, pi, cfg, S, P, T, Omega):
    """Probability-weighted loss of a simulated plan, objective (1) of the report."""
    total = 0.0
    for w in Omega:
        loss = 0.0
        for s, p, t in product(S, P, T):
            k = (s, p, t, w)
            loss += cfg["alpha"] * sim["U"][k] + cfg["beta"] * sim["W"][k]
            loss += (cfg["tau"] if t == T[-1] else cfg["h"]) * sim["L"][k]
        total += pi[w] * loss
    return total


def repair_plan(z_in, q_in, data, par):
    """Make a plan feasible for (K, C, R): keep the K largest products per store,
    clip deliveries to Q, then scale each day's deliveries down where regional
    supply or store capacity (worst scenario, including carried-over stock) binds."""
    S, P, T, Omega, d = (data[k] for k in ("S", "P", "T", "Omega", "d"))
    K, C, R, Q = par["K"], par["C"], par["R"], par["Q"]
    z, q = {}, {}
    for s in S:
        chosen = [p for p in P if int(z_in.get((s, p), 0)) == 1]
        chosen.sort(key=lambda p: -sum(q_in.get((s, p, t), 0.0) for t in T))
        keep = set(chosen[:K[s]])
        for p in P:
            z[s, p] = int(p in keep)
    for k in product(S, P, T):
        q[k] = min(max(q_in.get(k, 0.0), 0.0), Q[k]) if z[k[0], k[1]] else 0.0
    old = {(s, p, w): 0.0 for s, p, w in product(S, P, Omega)}
    for t in T:
        total = sum(q[s, p, t] for s, p in product(S, P))
        if total > R[t]:
            f = R[t] / total * (1 - 1e-9)
            for s, p in product(S, P):
                q[s, p, t] *= f
        for s in S:
            new_sum = sum(q[s, p, t] for p in P)
            worst_old = max(sum(old[s, p, w] for p in P) for w in Omega)
            room = max(C[s, t] - worst_old, 0.0)
            if new_sum > room:
                f = room / new_sum * (1 - 1e-9)
                for p in P:
                    q[s, p, t] *= f
        for s, p, w in product(S, P, Omega):
            D, o, new = d[s, p, t, w], old[s, p, w], q[s, p, t]
            sold_old, sales = min(o, D), min(o + new, D)
            old[s, p, w] = max(new - (sales - sold_old), 0.0)
    return z, q


def load_plan(run_dir):
    """Read z and q of a finished run (or of the Milestone 4 base_case folder)."""
    run_dir = Path(run_dir)
    a = pd.read_csv(run_dir / "assortment.csv")
    dl = pd.read_csv(run_dir / "deliveries.csv")
    day = {dt: i for i, dt in enumerate(DATES, 1)}
    z = {(int(r.store_id), int(r.product_id)): int(r.selected) for r in a.itertuples()}
    q = {(int(r.store_id), int(r.product_id), day[r.dt]): float(r.delivery) for r in dl.itertuples()}
    return z, q


# --------------------------------------------------------------------------
# Gurobi part
# --------------------------------------------------------------------------
def set_starts(m, v, data, par, plan):
    """Start 0 = zero-delivery plan (as in Milestone 4); start 1 = repaired warm plan."""
    from gurobipy import GRB  # noqa: F401  (imported for clarity of dependency)
    S, P, T, Omega, d = (data[k] for k in ("S", "P", "T", "Omega", "d"))
    z1, q1 = repair_plan(*plan, data, par)
    sim1 = simulate_plan(d, q1, S, P, T, Omega)
    m.NumStart = 2
    m.update()
    zero_q = {k: 0.0 for k in product(S, P, T)}
    starts = [({k: 0 for k in product(S, P)}, zero_q, simulate_plan(d, zero_q, S, P, T, Omega)),
              (z1, q1, sim1)]
    for number, (zs, qs, sim) in enumerate(starts):
        m.Params.StartNumber = number
        for k, x in v["z"].items():
            x.Start = zs[k]
        for k, x in v["q"].items():
            x.Start = qs[k]
        for name in (*STATES, "bA", "bO"):
            for k, x in v[name].items():
                x.Start = sim[name][k]
    m.update()
    return plan_objective(sim1, par["pi"], par["cfg"], S, P, T, Omega)


def fixed_lp_duals(m):
    """Duals of capacity/supply rows after fixing all binaries at the incumbent."""
    from gurobipy import GRB
    fm = m.fixed()
    try:
        fm.Params.OutputFlag = 0
        fm.optimize()
        if fm.Status != GRB.OPTIMAL:
            return None
        rows = []
        pat = re.compile(r"^(store_capacity|regional_supply)\[(.*)\]$")
        for c in fm.getConstrs():
            hit = pat.match(c.ConstrName)
            if not hit:
                continue
            idx = hit.group(2).split(",")
            if hit.group(1) == "store_capacity":
                rows.append(dict(constraint="store_capacity", store_id=int(idx[0]), day=int(idx[1]),
                                 scenario=idx[2], pi=float(c.Pi), slack=float(c.Slack), rhs=float(c.RHS)))
            else:
                rows.append(dict(constraint="regional_supply", store_id=np.nan, day=int(idx[0]),
                                 scenario="all", pi=float(c.Pi), slack=float(c.Slack), rhs=float(c.RHS)))
        return pd.DataFrame(rows)
    finally:
        fm.dispose()


def run_case(base_data, case, solver_settings, run_dir, warm_plan=None, duals=True):
    import gurobipy as gp
    from gurobipy import GRB
    import gurobi_model as gm

    run_dir.mkdir(parents=True, exist_ok=True)
    data = scaled_data(base_data, case["demand_scale"])
    par = gm.parameters(data, BASE_SETTINGS, case["overrides"])
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m, v = gm.build_model(data, par, env)
        try:
            start_obj = set_starts(m, v, data, par, warm_plan) if warm_plan else None
            m.Params.LogToConsole = 0
            m.Params.LogFile = str(run_dir / "gurobi.log")
            m.Params.OutputFlag = 1
            for name, value in solver_settings.items():
                m.setParam(name, value)
            t0 = time.time()
            m.optimize()
            if m.SolCount == 0:
                raise RuntimeError(f"{case['case_id']}: Gurobi status {m.Status}, no solution")
            res = gm.extract_and_audit(data, par, m, v)
            dual_table = fixed_lp_duals(m) if duals else None
            summary = dict(
                case_id=case["case_id"], group=case["group"], parameter=case["parameter"],
                value=case["value"], overrides=case["overrides"], demand_scale=case["demand_scale"],
                status="OPTIMAL_WITHIN_CONFIGURED_GAP" if m.Status == GRB.OPTIMAL
                else "FEASIBLE_INCUMBENT_NOT_PROVEN_OPTIMAL",
                gurobi_status_code=int(m.Status), objective=float(m.ObjVal),
                best_bound=float(m.ObjBound), mip_gap=float(m.MIPGap),
                runtime_seconds=float(m.Runtime), wall_seconds=time.time() - t0,
                warm_start_objective=start_obj, solver=solver_settings,
                model_settings={k: v for k, v in par["cfg"].items() if k != "probabilities"},
                scenario_weights=par["pi"], checks=res["checks"])
        finally:
            m.dispose()
    keep = ["assortment", "deliveries", "scenario_summary", "store_scenario_summary",
            "objective_components", "store_daily_capacity", "regional_daily_supply"]
    for name in keep:
        res[name].to_csv(run_dir / f"{name}.csv", index=False)
    if dual_table is not None:
        dual_table.to_csv(run_dir / "duals.csv", index=False)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def run_all(groups, time_limit, force=False, warm=True, duals=True, out_dir=OUT_DIR):
    import gurobi_model as gm

    data = gm.load_and_audit_inputs(DATA_DIR, STORES, PRODUCTS, DATES)
    solver = {**SOLVER_SETTINGS, "TimeLimit": time_limit}
    cases = build_cases(groups)
    runs = out_dir / "runs"
    print(f"{len(cases)} cases x {time_limit}s time limit (upper bound ~{len(cases)*time_limit/60:.0f} min)")
    base_plan = None
    for i, case in enumerate(cases, 1):
        run_dir = runs / case["case_id"]
        if (run_dir / "run_summary.json").exists() and not force:
            print(f"[{i}/{len(cases)}] {case['case_id']}: cached")
        else:
            if case["case_id"] == "base":
                warm_plan = load_plan(M4_BASE_DIR) if (warm and (M4_BASE_DIR / "deliveries.csv").exists()) else None
            else:
                warm_plan = base_plan if warm else None
            s = run_case(data, case, solver, run_dir, warm_plan, duals)
            print(f"[{i}/{len(cases)}] {case['case_id']}: obj={s['objective']:.4f} "
                  f"gap={100*s['mip_gap']:.2f}% time={s['runtime_seconds']:.0f}s")
        if case["case_id"] == "base":
            base_plan = load_plan(run_dir)


# --------------------------------------------------------------------------
# Analysis (pure pandas/matplotlib; works without Gurobi)
# --------------------------------------------------------------------------
def summarize_run(run_dir, base_assortment=None):
    meta = json.loads((run_dir / "run_summary.json").read_text())
    sc = pd.read_csv(run_dir / "scenario_summary.csv")
    pi = sc.probability
    E = lambda col: float((pi * sc[col]).sum())
    b = BASE_SETTINGS
    base_weight_loss = float((pi * (b["alpha"] * sc.unmet_demand + b["beta"] * sc.expired_stock
                                    + b["h"] * sc.overnight_carryover + b["tau"] * sc.day7_new_stock)).sum())
    cap = pd.read_csv(run_dir / "store_daily_capacity.csv")
    sup = pd.read_csv(run_dir / "regional_daily_supply.csv")
    asg = pd.read_csv(run_dir / "assortment.csv")
    tight = 1e-3
    cap["binding"] = cap.slack <= tight * cap.capacity
    row = dict(
        case_id=meta["case_id"], group=meta["group"], parameter=meta["parameter"], value=meta["value"],
        status=meta["status"], objective=meta["objective"], best_bound=meta["best_bound"],
        mip_gap=meta["mip_gap"], runtime_seconds=meta["runtime_seconds"],
        exp_demand=E("demand"), exp_sales=E("sales"), exp_fill_rate=E("sales") / E("demand"),
        exp_unmet=E("unmet_demand"), exp_unselected_demand=E("unselected_product_demand"),
        exp_selected_unmet=E("selected_product_unmet"), exp_expired=E("expired_stock"),
        exp_carryover=E("overnight_carryover"), exp_day7_stock=E("day7_new_stock"),
        base_weight_loss=base_weight_loss,
        weekly_delivery=float(pd.read_csv(run_dir / "deliveries.csv").delivery.sum()),
        products_selected=int(asg.selected.sum()),
        capacity_binding_store_days_any=int(cap.groupby(["store_id", "day"]).binding.any().sum()),
        capacity_binding_store_days_low=int(cap.loc[cap.scenario == "low"].binding.sum()),
        supply_binding_days=int((sup.slack <= tight * sup.limit).sum()),
        unused_supply=float(sup.slack.sum()),
    )
    for w in ("low", "typical", "high"):
        r = sc.loc[sc.scenario == w].iloc[0]
        row[f"fill_rate_{w}"] = float(r.sales / r.demand)
    if base_assortment is not None:
        merged = asg.merge(base_assortment, on=["store_id", "product_id"], suffixes=("", "_base"))
        row["assortment_changes_vs_base"] = int((merged.selected != merged.selected_base).sum())
    duals = run_dir / "duals.csv"
    if duals.exists():
        du = pd.read_csv(duals)
        row["capacity_marginal_value_total"] = float(-du.loc[du.constraint == "store_capacity", "pi"].sum())
        row["supply_marginal_value_total"] = float(-du.loc[du.constraint == "regional_supply", "pi"].sum())
    return row, sc.assign(case_id=meta["case_id"], group=meta["group"], value=meta["value"])


def analyze(out_dir=OUT_DIR):
    runs = out_dir / "runs"
    tables, figs = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    if not (runs / "base" / "run_summary.json").exists():
        raise FileNotFoundError("results/sensitivity/runs/base is missing; run the base case first")
    base_asg = pd.read_csv(runs / "base" / "assortment.csv")[["store_id", "product_id", "selected"]]
    rows, scen = [], []
    for run_dir in sorted(p for p in runs.iterdir() if (p / "run_summary.json").exists()):
        r, s = summarize_run(run_dir, base_asg)
        rows.append(r)
        scen.append(s)
    df = pd.DataFrame(rows)
    base = df.loc[df.case_id == "base"].iloc[0]

    df["d_fill_rate_pp"] = 100 * (df.exp_fill_rate - base.exp_fill_rate)
    df["d_unmet_pct"] = 100 * (df.exp_unmet / base.exp_unmet - 1)
    df["d_base_weight_loss_pct"] = 100 * (df.base_weight_loss / base.base_weight_loss - 1)
    df["d_objective_pct"] = 100 * (df.objective / base.objective - 1)
    # Objective values are comparable with the base only when the weights are unchanged.
    df["objective_comparable"] = ~df.group.isin(WEIGHT_GROUPS)
    # A change smaller than the larger of the two MIP gaps cannot be separated from solver tolerance.
    df["within_solver_tolerance"] = (df.d_objective_pct.abs() / 100
                                     < np.maximum(df.mip_gap, base.mip_gap)) & df.objective_comparable
    df.loc[df.case_id == "base", "within_solver_tolerance"] = False

    # Expand the base case into every group so each group's table/figure has its base point.
    present = [g for g in EXPERIMENTS if g in set(df.group)]
    expanded = [df.loc[df.group != "base"]]
    for g in present:
        key = EXPERIMENTS[g][0]
        expanded.append(df.loc[df.case_id == "base"].assign(group=g, parameter=key, value=BASE_VALUES[key]))
    long = pd.concat(expanded, ignore_index=True)
    long["value"] = long.value.astype(float)
    long["is_base"] = long.case_id == "base"
    long = long.sort_values(["group", "value"], key=lambda c: c.map(
        {g: i for i, g in enumerate(EXPERIMENTS)}) if c.name == "group" else c).reset_index(drop=True)

    df.to_csv(tables / "sensitivity_summary_by_case.csv", index=False)
    long.to_csv(tables / "sensitivity_summary_by_group.csv", index=False)
    pd.concat(scen, ignore_index=True).to_csv(tables / "sensitivity_by_scenario.csv", index=False)
    write_markdown(long, present, tables / "sensitivity_tables.md")
    write_duals(runs, tables)

    plot_metric_panels(long, present, "exp_fill_rate", "fill_rate", 100, "Expected fill rate (%)",
                       figs / "fig1_fill_rate_by_parameter.png",
                       "Fill rate under each tested setting (line: probability-weighted; band: low-to-high demand scenario)")
    plot_unmet_decomposition(long, present, figs / "fig2_unmet_decomposition.png")
    plot_metric_panels(long, present, "exp_expired", None, 1, "Expected expired stock (units)",
                       figs / "fig3_expired_by_parameter.png",
                       "Expired stock under each tested setting (probability-weighted)")
    plot_tornado(long, present, figs / "fig4_tornado.png")
    if (runs / "base" / "duals.csv").exists():
        plot_shadow_prices(pd.read_csv(runs / "base" / "duals.csv"), figs / "fig5_shadow_prices_base.png")
    print(f"Tables -> {tables}\nFigures -> {figs}")
    return df, long


def write_markdown(long, present, path):
    cols = [("value", "Value", "{:g}"), ("objective", "Objective", "{:,.1f}"),
            ("mip_gap", "Gap", "{:.2%}"), ("exp_fill_rate", "Fill rate", "{:.1%}"),
            ("exp_unmet", "Unmet", "{:.1f}"), ("exp_unselected_demand", "Not stocked", "{:.1f}"),
            ("exp_expired", "Expired", "{:.2f}"), ("exp_carryover", "Carryover", "{:.1f}"),
            ("weekly_delivery", "Delivered", "{:.1f}"), ("base_weight_loss", "Loss @ base wts", "{:,.1f}"),
            ("assortment_changes_vs_base", "Assortment changes", "{:d}")]
    lines = ["# Sensitivity analysis tables", "",
             "All quantities are probability-weighted over the low/typical/high scenarios and in "
             "normalized units. 'Not stocked' is demand for products outside the assortment "
             "(part of Unmet). 'Loss @ base wts' re-scores each plan with alpha=5, beta=1, h=0.2, tau=0.5. "
             "Rows marked * are the base case. Objective changes smaller than the MIP gaps are marked (~).", ""]
    for g in present:
        sub = long.loc[long.group == g]
        lines += [f"## {EXPERIMENTS[g][2]}", "",
                  "| " + " | ".join(c[1] for c in cols) + " |",
                  "|" + "---|" * len(cols)]
        for r in sub.itertuples():
            cells = []
            for c, _, f in cols:
                val = getattr(r, c)
                cells.append("" if pd.isna(val) else f.format(int(val) if f == "{:d}" else val))
            cells[0] += "*" if r.is_base else ""
            if r.within_solver_tolerance and not r.is_base:
                cells[1] += " (~)"
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def write_duals(runs, tables):
    frames = []
    for run_dir in sorted(runs.iterdir()):
        f = run_dir / "duals.csv"
        if f.exists():
            du = pd.read_csv(f)
            cap = (du.loc[du.constraint == "store_capacity"].groupby(["store_id", "day"], as_index=False)
                   .agg(pi_sum=("pi", "sum"), min_slack=("slack", "min")))
            cap["constraint"] = "store_capacity"
            sup = du.loc[du.constraint == "regional_supply", ["day", "pi", "slack"]].rename(
                columns={"pi": "pi_sum", "slack": "min_slack"})
            sup["constraint"] = "regional_supply"
            out = pd.concat([cap, sup], ignore_index=True)
            # Loss reduction per extra unit of capacity/supply (pi <= 0 for a <= row in a min problem).
            out["marginal_value"] = -out.pi_sum
            frames.append(out.assign(case_id=run_dir.name))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(tables / "shadow_prices.csv", index=False)


# ----------------------------- figures -----------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 10,
        "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "figure.dpi": 150,
        "savefig.bbox": "tight", "legend.frameon": False})
    return plt


def _grid(plt, n):
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def plot_metric_panels(long, present, metric, band_prefix, scale, ylabel, path, title):
    plt = _style()
    fig, axes = _grid(plt, len(present))
    for ax, g in zip(axes, present):
        sub = long.loc[long.group == g]
        x = np.arange(len(sub))
        y = scale * sub[metric].to_numpy()
        if band_prefix:
            lo = scale * sub[f"{band_prefix}_high"].to_numpy()  # high demand -> lowest fill rate
            hi = scale * sub[f"{band_prefix}_low"].to_numpy()
            ax.fill_between(x, lo, hi, color=BLUE, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=5, mec="white", mew=1.5, zorder=3)
        b = np.flatnonzero(sub.is_base.to_numpy())
        if len(b):
            ax.plot(x[b], y[b], marker="o", ms=8, mfc="white", mec=BLUE, mew=2, ls="none", zorder=4)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x, [fmt_value(v) for v in sub.value])
        ax.set_xlabel(EXPERIMENTS[g][2])
        ax.set_ylabel(ylabel)
        ax.margins(x=0.12, y=0.18)
    fig.suptitle(title + "\nOpen marker = base case", color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_unmet_decomposition(long, present, path):
    plt = _style()
    fig, axes = _grid(plt, len(present))
    for ax, g in zip(axes, present):
        sub = long.loc[long.group == g]
        x = np.arange(len(sub))
        a = sub.exp_unselected_demand.to_numpy()
        b = sub.exp_selected_unmet.to_numpy()
        ax.bar(x, a, width=0.6, color=ORANGE, edgecolor="white", linewidth=1, label="Demand for products not stocked")
        ax.bar(x, b, width=0.6, bottom=a, color=BLUE, edgecolor="white", linewidth=1,
               label="Shortage on stocked products")
        for xi, tot in zip(x, a + b):
            ax.annotate(f"{tot:.0f}", (xi, tot), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x, [fmt_value(v) + ("*" if base else "") for v, base in zip(sub.value, sub.is_base)])
        ax.set_xlabel(EXPERIMENTS[g][2])
        ax.set_ylabel("Expected unmet demand (units)")
        ax.margins(y=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, bbox_to_anchor=(0.02, 0.995))
    fig.suptitle("Where unmet demand comes from (* = base case)", color=INK, fontsize=10,
                 x=0.02, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tornado(long, present, path):
    plt = _style()
    metrics = [("d_fill_rate_pp", "Change in expected fill rate (percentage points)"),
               ("d_base_weight_loss_pct", "Change in expected loss at base weights (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(present) + 1.4))
    for ax, (m, label) in zip(axes, metrics):
        spans = []
        for g in present:
            sub = long.loc[long.group == g]
            lo, hi = sub.loc[sub[m].idxmin()], sub.loc[sub[m].idxmax()]
            spans.append((g, lo[m], hi[m], lo.value, hi.value))
        spans.sort(key=lambda r: r[2] - r[1])
        width = max([abs(r[1]) for r in spans] + [abs(r[2]) for r in spans] + [1e-9])
        for i, (g, lo, hi, vlo, vhi) in enumerate(spans):
            if hi - lo < 1e-3 * width:
                ax.annotate("no change", (0, i), textcoords="offset points", xytext=(4, 0),
                            ha="left", va="center", fontsize=7.5, color=INK2)
                continue
            ax.barh(i, hi - lo, left=lo, height=0.55, color=BLUE, alpha=0.85)
            ax.annotate(fmt_value(vlo), (lo, i), textcoords="offset points", xytext=(-4, 0),
                        ha="right", va="center", fontsize=7.5, color=INK2)
            ax.annotate(fmt_value(vhi), (hi, i), textcoords="offset points", xytext=(4, 0),
                        ha="left", va="center", fontsize=7.5, color=INK2)
        lo_all = min(min(r[1] for r in spans), 0)
        hi_all = max(max(r[2] for r in spans), 0)
        pad = 0.15 * (hi_all - lo_all or 1)
        ax.set_xlim(lo_all - pad, hi_all + pad)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(spans)), [EXPERIMENTS[s[0]][2] for s in spans])
        ax.set_ylim(-0.7, len(spans) - 0.3)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Which assumptions move the results most (labels = tested value at each end)",
                 color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_shadow_prices(du, path):
    plt = _style()
    cap = du.loc[du.constraint == "store_capacity"].copy()
    cap["store_id"] = cap.store_id.astype(int)
    cap = (cap.groupby(["store_id", "day"]).pi.sum() * -1).unstack(0)
    sup = -du.loc[du.constraint == "regional_supply"].set_index("day").pi
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    width = 0.26
    for j, (s, color) in enumerate(zip(cap.columns, (BLUE, ORANGE, AQUA))):
        axes[0].bar(cap.index + (j - 1) * width, cap[s], width=width, color=color,
                    edgecolor="white", linewidth=1, label=f"Store {s}")
    axes[0].set_xlabel("Day (1 = June 26)")
    axes[0].set_ylabel("Loss reduction per extra unit")
    axes[0].set_title("Store capacity", loc="left")
    axes[0].legend(ncol=3, loc="upper left")
    axes[0].margins(y=0.2)
    axes[1].bar(sup.index, sup.values, width=0.6, color=BLUE)
    axes[1].set_xlabel("Day (1 = June 26)")
    axes[1].set_ylabel("Loss reduction per extra unit")
    axes[1].set_title("Regional supply", loc="left")
    fig.suptitle("Shadow prices in the base plan (assortment fixed; local marginal values)",
                 color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, choices=list(EXPERIMENTS))
    ap.add_argument("--time-limit", type=float, default=SENSITIVITY_TIME_LIMIT,
                    help="seconds per case (default 60)")
    ap.add_argument("--force", action="store_true", help="re-solve cases that are already cached")
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--no-duals", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run_all(args.groups, args.time_limit, args.force, not args.no_warm_start, not args.no_duals)
    analyze()


if __name__ == "__main__":
    main()
