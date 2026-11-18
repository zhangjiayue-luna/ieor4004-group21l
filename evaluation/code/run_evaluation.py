"""Group 21L, Milestone 5: build all plans and score them on common demand paths.

Usage:
    python run_evaluation.py --repo <ieor4004-group21l checkout> --eval <dir with eval CSVs>
                             --out <output dir> [--time-limit 900]

Plans
  B-raw / B-rec : rule-based baseline on raw sales / stockout-adjusted demand
  M-raw / M-rec : Milestone 4 MILP on raw / adjusted scenarios (HiGHS, same settings)
  M-rec-Gurobi  : the official Milestone 4 Gurobi incumbent (results/base_case)

Demand paths used for scoring (identical for every plan)
  scenarios      : the three adjusted planning scenarios, probability-weighted
  holdout        : stockout-adjusted demand for June 26-July 2 (training-period references)
  holdout_plus25 : same, with estimated lost demand increased by 25%
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as ev  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--time-limit", type=float, default=900)
args = ap.parse_args()
repo, evd, out = Path(args.repo), Path(args.eval), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- inputs
hist = pd.read_csv(repo / "data/recovered_demand.csv")
sc = pd.read_csv(repo / "data/scenarios_1827.csv")
fc_file = ev.to_array(pd.read_csv(repo / "data/point_forecast_609.csv"), "fc")
pi = sc.groupby("scenario").weight.first().to_dict()
scen_rec = {w: ev.to_array(sc[sc.scenario == w], "demand") for w in ev.OMEGA}
kappa = {w: float(np.mean(scen_rec[w][fc_file > 0] / fc_file[fc_file > 0])) for w in ev.OMEGA}

inputs = {}
for tag, col in (("rec", "demand"), ("raw", "sale_amount")):
    fc = ev.point_forecast(hist, col)
    inputs[tag] = dict(fc=fc, totals=ev.ranking_totals(hist, col),
                       scen={w: kappa[w] * fc for w in ev.OMEGA})
assert np.abs(inputs["rec"]["fc"] - fc_file).max() < 1e-9, "forecast rule mismatch"
assert max(np.abs(inputs["rec"]["scen"][w] - scen_rec[w]).max() for w in ev.OMEGA) < 1e-9

hold = {"holdout": ev.to_array(pd.read_csv(evd / "eval_recovered_demand_609.csv"), "demand"),
        "holdout_plus25": ev.to_array(pd.read_csv(evd / "eval_recovered_demand_plus25_609.csv"), "demand")}
observed = ev.to_array(pd.read_csv(evd / "eval_recovered_demand_609.csv"), "sale_amount")

# raw-input scenarios, saved so the team can re-solve M-raw with Gurobi
rows = []
for w in ev.OMEGA:
    for i, s in enumerate(ev.STORES):
        for j, p in enumerate(ev.PRODUCTS):
            for k, dt in enumerate(ev.DATES):
                rows.append(dict(store_id=s, product_id=p, t=k + 1, dt=dt, scenario=w,
                                 weight=pi[w], demand=inputs["raw"]["scen"][w][i, j, k]))
pd.DataFrame(rows).to_csv(out / "scenarios_raw_sales_1827.csv", index=False)

# ---------------------------------------------------------------- plans
plans, solver = {}, {}
for tag in ("raw", "rec"):
    inp = inputs[tag]
    q, z = ev.baseline_plan(inp["totals"], inp["fc"], inp["scen"])
    ev.check_feasible(q, z, inp["scen"])
    plans[f"B-{tag}"] = (q, z)
    cache = out / f"milp_{tag}.npz"
    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        q, z, info = npz["q"], npz["z"], npz["info"].item()
    else:
        q, z, info = ev.solve_milp(inp["scen"], pi, time_limit=args.time_limit,
                                   log=out / f"highs_{tag}.log")
        np.savez(cache, q=q, z=z, info=np.array(info, dtype=object))
    ev.check_feasible(q, z, inp["scen"])
    plans[f"M-{tag}"] = (q, z)
    solver[f"M-{tag}"] = info
q, z = ev.plan_from_repo(repo / "results/base_case")
ev.check_feasible(q, z, scen_rec)
plans["M-rec-Gurobi"] = (q, z)
solver["M-rec-Gurobi"] = json.loads((repo / "results/base_case/solver_summary.json").read_text())

# ---------------------------------------------------------------- scoring
cap = np.array([ev.BASE["C"][s] for s in ev.STORES])[:, None]
records, store_records = [], []
for name, (q, z) in plans.items():
    paths = {**{f"scenario_{w}": scen_rec[w] for w in ev.OMEGA}, **hold}
    for path, d in paths.items():
        m = ev.metrics(q, z, d)
        m["capacity_overflow"] = float(max(0, (ev.simulate(q, d)["V"].sum(axis=1) - cap).max()))
        records.append(dict(policy=name, path=path, **m))
        for i, s in enumerate(ev.STORES):
            sm = ev.metrics(q[i:i + 1], z[i:i + 1], d[i:i + 1])
            store_records.append(dict(policy=name, path=path, store_id=s, **sm))
res = pd.DataFrame(records)
exp = (res[res.path.str.startswith("scenario_")]
       .assign(wt=lambda x: x.path.str.replace("scenario_", "").map(pi)))
num = [c for c in res.columns if c not in ("policy", "path")]
expected = exp.groupby("policy").apply(
    lambda g: pd.Series({c: (g[c] * g.wt).sum() for c in num}), include_groups=False).reset_index()
expected["fill_rate"] = (exp.assign(a=exp.sales * exp.wt, d=exp.demand * exp.wt)
                         .groupby("policy").apply(lambda g: g.a.sum() / g.d.sum(),
                                                  include_groups=False).values)
expected["path"] = "scenarios_expected"
res = pd.concat([res, expected[res.columns]], ignore_index=True)
res.to_csv(out / "evaluation_all.csv", index=False)
pd.DataFrame(store_records).to_csv(out / "evaluation_by_store.csv", index=False)

plan_rows = pd.concat([ev.plan_to_long(q, z, n) for n, (q, z) in plans.items()])
plan_rows.to_csv(out / "plans_long.csv", index=False)
(out / "solver_info.json").write_text(json.dumps(
    {k: {kk: (vv if isinstance(vv, (int, float, str, list, dict)) else str(vv))
         for kk, vv in v.items()} for k, v in solver.items()}, indent=2, default=str))
(out / "run_info.json").write_text(json.dumps(dict(
    kappa=kappa, pi=pi, base=dict(ev.BASE, C={str(k): v for k, v in ev.BASE["C"].items()}),
    holdout_total=float(hold["holdout"].sum()), holdout_plus25_total=float(hold["holdout_plus25"].sum()),
    holdout_observed_sales=float(observed.sum()),
    raw_point_forecast_total=float(inputs["raw"]["fc"].sum()),
    rec_point_forecast_total=float(inputs["rec"]["fc"].sum())), indent=2))
print(res[res.path.isin(["scenarios_expected", "holdout", "holdout_plus25"])]
      [["policy", "path", "fill_rate", "unmet", "unmet_unselected", "expired", "carryover",
        "day7_stock", "loss", "capacity_overflow"]].round(3).to_string(index=False))
print(json.dumps(solver, indent=1, default=str)[:1500])
