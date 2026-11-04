"""Group 21L: optimization model and solution verification."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from itertools import product
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

STATES = ("O", "V", "A", "Sold", "Snew", "L", "W", "U")


def load_and_audit_inputs(data_dir, stores, products, dates):
    files = {"forecast": Path(data_dir) / "point_forecast_609.csv",
             "scenarios": Path(data_dir) / "scenarios_1827.csv",
             "history": Path(data_dir) / "recovered_demand.csv"}
    frames = {key: pd.read_csv(path) for key, path in files.items()}
    f, sc, hist = (frames[k] for k in ("forecast", "scenarios", "history"))
    required = {"forecast": {"store_id", "product_id", "t", "dt", "fc"},
                "scenarios": {"store_id", "product_id", "t", "dt", "scenario", "weight", "demand"},
                "history": {"store_id", "product_id", "dt", "sale_amount", "demand"}}
    for key, frame in frames.items():
        if not required[key].issubset(frame.columns) or frame[list(required[key])].isna().any().any():
            raise ValueError(f"Missing columns/values in {files[key].name}")
        for col in ("store_id", "product_id"):
            v = pd.to_numeric(frame[col], errors="raise")
            if not np.isfinite(v).all() or not (v == np.floor(v)).all():
                raise ValueError(f"Noninteger ID in {files[key].name}")
            frame[col] = v.astype(int)
        frame["dt"] = pd.to_datetime(frame["dt"], errors="raise").dt.strftime("%Y-%m-%d")
    sc["scenario"] = sc.scenario.astype(str).str.strip().str.lower()
    omega = ["low", "typical", "high"]
    expected3 = set(product(stores, products, dates))
    expected4 = set(product(stores, products, dates, omega))
    for frame, columns, expected in [(f, ["store_id", "product_id", "dt"], expected3),
                                     (sc, ["store_id", "product_id", "dt", "scenario"], expected4)]:
        if frame.duplicated(columns).any() or set(frame[columns].itertuples(index=False, name=None)) != expected:
            raise ValueError("Duplicate, missing, or extra forecast/scenario keys")
        expected_t = frame.dt.map({day: i for i, day in enumerate(dates, 1)})
        if not (pd.to_numeric(frame.t, errors="raise") == expected_t).all():
            raise ValueError("t does not match dt")
    for frame, columns in [(f, ["fc"]), (sc, ["demand", "weight"]),
                           (hist, ["sale_amount", "demand"])]:
        for col in columns:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if not np.isfinite(frame[col]).all() or (frame[col] < 0).any():
                raise ValueError(f"Invalid values in {col}")
    hist_dates = pd.date_range("2024-03-28", "2024-06-25").strftime("%Y-%m-%d").tolist()
    hist_keys = ["store_id", "product_id", "dt"]
    if hist.duplicated(hist_keys).any() or set(hist[hist_keys].itertuples(index=False, name=None)) != set(product(stores, products, hist_dates)):
        raise ValueError("Historical recovered-demand coverage differs from the 90-day sample")
    if (hist.demand < hist.sale_amount - 1e-9).any():
        raise ValueError("Recovered demand is below observed sales")
    pi = {}
    for w in omega:
        weights = sc.loc[sc.scenario == w, "weight"].unique()
        if len(weights) != 1:
            raise ValueError("Each scenario must have one consistent weight")
        pi[w] = float(weights[0])
    if not math.isclose(sum(pi.values()), 1, abs_tol=1e-9):
        raise ValueError("Scenario probabilities do not sum to one")
    # File values remain authoritative; this checks the agreed common-multiplier structure.
    merged = sc.merge(f[["store_id", "product_id", "dt", "fc"]],
                      on=["store_id", "product_id", "dt"], validate="many_to_one")
    config_rows = []
    for w in omega:
        sub = merged[merged.scenario == w]
        ratios = sub.loc[sub.fc > 0, "demand"] / sub.loc[sub.fc > 0, "fc"]
        kappa = float(ratios.mean()) if len(ratios) else np.nan
        multiplier_error = float((sub.demand - sub.fc * kappa).abs().max())
        if not math.isfinite(kappa) or multiplier_error > 1e-8:
            raise ValueError("Scenario file does not follow a common forecast multiplier")
        config_rows.append(dict(scenario=w, probability=pi[w], multiplier=kappa,
                                total_demand=float(sub.demand.sum()),
                                max_multiplier_check_error=multiplier_error))
    # Independently check that fc follows the submitted seven-day level × store weekday index.
    recent = hist[hist.dt.between("2024-06-19", "2024-06-25")].groupby(["store_id", "product_id"]).demand.mean()
    totals = hist[hist.dt.between("2024-05-29", "2024-06-25")].groupby(["store_id", "dt"], as_index=False).demand.sum()
    totals["weekday"] = pd.to_datetime(totals.dt).dt.dayofweek
    store_mean = totals.groupby("store_id").demand.mean()
    weekday_mean = totals.groupby(["store_id", "weekday"]).demand.mean()
    implied = [recent.loc[(r.store_id, r.product_id)] *
               weekday_mean.loc[(r.store_id, pd.Timestamp(r.dt).dayofweek)] / store_mean.loc[r.store_id]
               for r in f.itertuples()]
    forecast_method_difference = float(np.max(np.abs(np.asarray(implied) - f.fc.to_numpy())))
    audit = pd.DataFrame([{"file": path.name, "rows": len(frames[key]),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for key, path in files.items()])
    d = {(int(r.store_id), int(r.product_id), int(r.t), r.scenario): float(r.demand)
         for r in sc.itertuples()}
    return dict(S=list(stores), P=list(products), T=list(range(1, 8)), Omega=omega,
                dates=dict(enumerate(dates, 1)), d=d, pi=pi, scenario_configuration=pd.DataFrame(config_rows),
                input_audit=audit, forecast_method_max_difference=forecast_method_difference,
                forecast=f, scenarios=sc, history=hist)


def parameters(data, settings, overrides=None):
    cfg = copy.deepcopy(settings)
    overrides = {} if overrides is None else dict(overrides)
    if set(overrides) - set(cfg):
        raise ValueError("Unknown parameter setting")
    cfg.update(overrides)
    S, P, T = (data[k] for k in ("S", "P", "T"))
    K = {s: int(cfg["K"]) for s in S}
    if cfg["K"] != int(cfg["K"]) or not 0 <= cfg["K"] <= len(P):
        raise ValueError("Invalid assortment cap")
    C = {(s, t): float(cfg["capacity_by_store"][s]) * cfg["capacity_scale"] for s in S for t in T}
    R = {t: float(cfg["regional_supply"]) * cfg["supply_scale"] for t in T}
    if any(not math.isfinite(v) or v <= 0 for v in [*C.values(), *R.values()]):
        raise ValueError("Invalid resource limits")
    if any(not math.isfinite(cfg[k]) or cfg[k] < 0 for k in ("alpha", "beta", "h", "tau")):
        raise ValueError("Invalid penalty weight")
    pi = cfg["probabilities"] if cfg["probabilities"] is not None else data["pi"]
    if set(pi) != set(data["Omega"]) or any(not math.isfinite(v) or v < 0 for v in pi.values()) or not math.isclose(sum(pi.values()), 1, abs_tol=1e-9):
        raise ValueError("Invalid scenario weights")
    Q = {(s, p, t): min(C[s, t], R[t]) for s, p, t in product(S, P, T)}
    B = {k: max(C[k[0], k[2]], value) for k, value in data["d"].items()}
    return dict(C=C, R=R, K=K, Q=Q, B=B, pi=pi, cfg=cfg)


def build_model(data, par, env):
    S, P, T, Omega = (data[k] for k in ("S", "P", "T", "Omega"))
    k3, k4 = list(product(S, P, T)), list(product(S, P, T, Omega))
    d, C, Q, B, cfg = data["d"], par["C"], par["Q"], par["B"], par["cfg"]
    m = gp.Model("Group21L_Milestone4", env=env)
    v = {"z": m.addVars(S, P, vtype=GRB.BINARY, name="z"),
         "q": m.addVars(k3, lb=0, ub=Q, name="q")}
    for name in STATES:
        ub = {k: d[k] if name == "U" else min(C[k[0], k[2]], d[k])
              if name in ("A", "Sold") else C[k[0], k[2]] for k in k4}
        v[name] = m.addVars(k4, lb=0, ub=ub, name=name)
    v["bA"] = m.addVars(k4, vtype=GRB.BINARY, name="bA")
    v["bO"] = m.addVars(k4, vtype=GRB.BINARY, name="bO")
    z, q = v["z"], v["q"]
    O, V, A, Sold, Snew, L, W, U = (v[n] for n in STATES)
    # (5) Weekly assortment and delivery eligibility.
    m.addConstrs((gp.quicksum(z[s, p] for p in P) <= par["K"][s] for s in S), "assortment")
    m.addConstrs((q[k] <= Q[k] * z[k[0], k[1]] for k in k3), "delivery_selection")
    # (6) Initial old stock = 0; new stock carries for one night only.
    for s, p, t, w in k4:
        m.addConstr(O[s, p, t, w] == (0 if t == T[0] else L[s, p, t-1, w]),
                    name=f"old_stock[{s},{p},{t},{w}]")
    # (7) Inventory before sales.
    m.addConstrs((V[k] == O[k] + q[k[:3]] for k in k4), "inventory_balance")
    # (8) Capacity includes old stock and new delivery under every scenario.
    m.addConstrs((gp.quicksum(V[s, p, t, w] for p in P) <= C[s, t]
                  for s in S for t in T for w in Omega), "store_capacity")
    # (9) Shared regional limit on new deliveries.
    m.addConstrs((gp.quicksum(q[s, p, t] for s in S for p in P) <= par["R"][t] for t in T), "regional_supply")
    # (10) Exact total sales A = min(V, d).
    m.addConstrs((A[k] <= V[k] for k in k4), "sales_inventory")
    m.addConstrs((A[k] <= d[k] for k in k4), "sales_demand")
    m.addConstrs((A[k] >= V[k] - B[k] * v["bA"][k] for k in k4), "sales_branch1")
    m.addConstrs((A[k] >= d[k] - B[k] * (1-v["bA"][k]) for k in k4), "sales_branch2")
    # (11) Exact old-stock sales Sold = min(O, d), enforcing FIFO.
    m.addConstrs((Sold[k] <= O[k] for k in k4), "old_sales_inventory")
    m.addConstrs((Sold[k] <= d[k] for k in k4), "old_sales_demand")
    m.addConstrs((Sold[k] >= O[k] - B[k] * v["bO"][k] for k in k4), "old_sales_branch1")
    m.addConstrs((Sold[k] >= d[k] - B[k] * (1-v["bO"][k]) for k in k4), "old_sales_branch2")
    # (12) New-stock sales and carryover.
    m.addConstrs((Snew[k] == A[k]-Sold[k] for k in k4), "new_sales")
    m.addConstrs((L[k] == q[k[:3]]-Snew[k] for k in k4), "new_left")
    # (13) Old-stock expiration and unmet demand, including unselected products.
    m.addConstrs((W[k] == O[k]-Sold[k] for k in k4), "expired_old")
    m.addConstrs((U[k] == d[k]-A[k] for k in k4), "unmet")
    # (4) Day-7 remaining new stock is charged tau, not h or expired waste.
    objective = gp.quicksum(par["pi"][w] * (
        gp.quicksum(cfg["alpha"] * U[s,p,t,w] + cfg["beta"] * W[s,p,t,w] for s,p,t in k3)
        + cfg["h"] * gp.quicksum(L[s,p,t,w] for s in S for p in P for t in T[:-1])
        + cfg["tau"] * gp.quicksum(L[s,p,T[-1],w] for s in S for p in P)) for w in Omega)
    m.setObjective(objective, GRB.MINIMIZE)
    # Complete feasible zero-delivery MIP start; it adds no constraint or cost.
    for variables in v.values():
        for x in variables.values():
            x.Start = 0
    for k in k4:
        U[k].Start = d[k]
    m.update()
    return m, v


def extract_and_audit(data, par, m, v, tolerance=1e-5):
    S, P, T, Omega = (data[k] for k in ("S", "P", "T", "Omega"))
    cfg = par["cfg"]
    q = {k: float(x.X) for k, x in v["q"].items()}
    z = {k: int(round(x.X)) for k, x in v["z"].items()}
    assortment = pd.DataFrame([dict(store_id=s, product_id=p, selected=z[s,p],
                                    weekly_delivery=sum(q[s,p,t] for t in T)) for s,p in product(S,P)])
    deliveries = pd.DataFrame([dict(store_id=s, product_id=p, dt=data["dates"][t], delivery=q[s,p,t])
                               for s,p,t in product(S,P,T)])
    rows, state_error = [], 0.0
    for s,p,w in product(S,P,Omega):
        old = 0.0
        for t in T:
            k=(s,p,t,w)
            D, new = data["d"][k], q[s,p,t]
            avail = old + new
            sold_old, sales = min(old,D), min(avail,D)
            sold_new = sales-sold_old
            left = new-sold_new
            simulated = dict(O=old,V=avail,A=sales,Sold=sold_old,Snew=sold_new,L=left,W=old-sold_old,U=D-sales)
            actual = {n: float(v[n][k].X) for n in STATES}
            state_error=max(state_error,max(abs(actual[n]-simulated[n]) for n in STATES))
            rows.append(dict(store_id=s,product_id=p,dt=data["dates"][t],day=t,scenario=w,
                             demand=D,delivery=new,selected=z[s,p],**actual))
            old=left
    outcomes=pd.DataFrame(rows)
    capacity=outcomes.groupby(["store_id","dt","day","scenario"],as_index=False).V.sum()
    capacity["capacity"]=[par["C"][s,t] for s,t in zip(capacity.store_id,capacity.day)]
    capacity["slack"]=capacity.capacity-capacity.V
    capacity["utilization"]=capacity.V/capacity.capacity
    supply=deliveries.groupby("dt",as_index=False).delivery.sum()
    supply["limit"]=[par["R"][t] for t in T]
    supply["slack"]=supply.limit-supply.delivery
    checks=dict(max_constraint_violation=float(m.ConstrVio),max_bound_violation=float(m.BoundVio),
                max_integrality_violation=float(m.IntVio),max_independent_inventory_error=state_error,
                max_capacity_violation=max(0.0,-float(capacity.slack.min())),
                max_supply_violation=max(0.0,-float(supply.slack.min())),
                max_assortment_violation=max(max(0,sum(z[s,p] for p in P)-par["K"][s]) for s in S),
                max_delivery_selection_violation=max(max(0,q[s,p,t]-par["Q"][s,p,t]*z[s,p]) for s,p,t in product(S,P,T)))
    if max(checks.values())>tolerance:
        raise RuntimeError(f"Solution audit failed: {checks}")
    scenario_rows,store_rows,component_rows=[],[],[]
    for w in Omega:
        sub=outcomes[outcomes.scenario==w]
        D,sales,unmet,expired=map(float,(sub.demand.sum(),sub.A.sum(),sub.U.sum(),sub.W.sum()))
        carry=float(sub.loc[sub.day<T[-1],"L"].sum())
        terminal=float(sub.loc[sub.day==T[-1],"L"].sum())
        omitted=float(sub.loc[sub.selected==0,"demand"].sum())
        comp=dict(unmet_demand=cfg["alpha"]*unmet,expired_stock=cfg["beta"]*expired,
                  overnight_carryover=cfg["h"]*carry,terminal_new_stock=cfg["tau"]*terminal)
        for label,value in comp.items():
            component_rows.append(dict(scenario=w,component=label,loss=value,probability=par["pi"][w],weighted_loss=value*par["pi"][w]))
        scenario_rows.append(dict(scenario=w,probability=par["pi"][w],demand=D,sales=sales,fill_rate=sales/D if D else np.nan,
                                  unmet_demand=unmet,unselected_product_demand=omitted,selected_product_unmet=unmet-omitted,
                                  expired_stock=expired,overnight_carryover=carry,day7_new_stock=terminal,loss=sum(comp.values())))
        for s in S:
            ss=sub[sub.store_id==s]
            total,served=float(ss.demand.sum()),float(ss.A.sum())
            store_rows.append(dict(store_id=s,scenario=w,demand=total,sales=served,fill_rate=served/total if total else np.nan,
                                   unmet_demand=float(ss.U.sum()),expired_stock=float(ss.W.sum()),
                                   carryover=float(ss.loc[ss.day<T[-1],"L"].sum()),day7_new_stock=float(ss.loc[ss.day==T[-1],"L"].sum())))
    components=pd.DataFrame(component_rows)
    checks["objective_reconciliation_error"]=abs(float(components.weighted_loss.sum())-float(m.ObjVal))
    if checks["objective_reconciliation_error"]>tolerance*max(1,abs(m.ObjVal)):
        raise RuntimeError("Objective components do not reconcile")
    return dict(assortment=assortment,deliveries=deliveries,scenario_outcomes=outcomes,
                scenario_summary=pd.DataFrame(scenario_rows),store_scenario_summary=pd.DataFrame(store_rows),
                objective_components=components,store_daily_capacity=capacity,regional_daily_supply=supply,checks=checks)


def solve_case(data, settings, solver_settings, output_dir, overrides=None):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    par=parameters(data,settings,overrides)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag",0)
        env.start()
        m,v=build_model(data,par,env)
        try:
            m.Params.LogToConsole=0
            m.Params.LogFile=str(output_dir/"gurobi.log")
            m.Params.OutputFlag=1
            for name,value in solver_settings.items():
                m.setParam(name,value)
            m.write(str(output_dir/"model.lp"))
            m.optimize()
            if m.SolCount==0:
                if m.Status==GRB.INFEASIBLE:
                    m.computeIIS()
                    m.write(str(output_dir/"infeasible.ilp"))
                raise RuntimeError(f"Gurobi status {m.Status}; no candidate solution")
            result=extract_and_audit(data,par,m,v)
            summary=dict(input_source="forecast_team_csv",
                         status="OPTIMAL_WITHIN_CONFIGURED_GAP" if m.Status==GRB.OPTIMAL else "FEASIBLE_INCUMBENT_NOT_PROVEN_OPTIMAL",
                         gurobi_status_code=int(m.Status),objective_weighted_loss=float(m.ObjVal),best_bound=float(m.ObjBound),
                         mip_gap=float(m.MIPGap),configured_mip_gap=float(m.Params.MIPGap),runtime_seconds=float(m.Runtime),
                         variables=int(m.NumVars),binary_variables=int(m.NumBinVars),constraints=int(m.NumConstrs),
                         gurobi_version=list(gp.gurobi.version()),checks=result["checks"],
                         selected_products_by_store={str(s):int(result["assortment"].loc[result["assortment"].store_id==s,"selected"].sum()) for s in data["S"]})
            for name,table in result.items():
                if isinstance(table,pd.DataFrame): table.to_csv(output_dir/f"{name}.csv",index=False)
            data["input_audit"].to_csv(output_dir/"input_audit.csv",index=False)
            data["scenario_configuration"].to_csv(output_dir/"scenario_configuration.csv",index=False)
            (output_dir/"solver_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            (output_dir/"run_parameters.json").write_text(json.dumps(dict(model=par["cfg"],scenario_weights=par["pi"],solver=solver_settings),indent=2)+"\n")
            m.write(str(output_dir/"solution.sol"))
        finally:
            m.dispose()
    result["solver_summary"]=summary
    return result
