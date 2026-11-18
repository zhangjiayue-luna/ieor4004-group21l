"""Group 21L, Milestone 5: rule-based baseline and common evaluation.

Everything here follows the Milestone 4 model exactly:
two-day sellable life, FIFO issuing, zero initial old stock, sales = min(stock, demand),
lost (not backordered) unmet demand, store capacity on old + new stock, regional limit
on new deliveries, and loss weights alpha, beta, h, tau.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

STORES = [18, 145, 235]
PRODUCTS = [21, 70, 79, 94, 117, 118, 136, 140, 207, 213, 215, 247, 249, 296, 366, 422,
            496, 500, 554, 556, 633, 634, 636, 638, 717, 738, 775, 797, 843]
DATES = pd.date_range("2024-06-26", periods=7).strftime("%Y-%m-%d").tolist()
T = list(range(1, 8))
OMEGA = ["low", "typical", "high"]
BASE = dict(K=20, C={18: 51.25, 145: 46.93, 235: 46.41}, R=144.43,
            alpha=5.0, beta=1.0, h=0.2, tau=0.5)
NS, NP, NT = len(STORES), len(PRODUCTS), len(T)


# ----------------------------------------------------------------------------- data
def to_array(df, value, keys=("store_id", "product_id", "dt")):
    """Long table -> array [store, product, day] in STORES/PRODUCTS/DATES order."""
    idx = pd.MultiIndex.from_product([STORES, PRODUCTS, DATES], names=list(keys))
    s = df.set_index(list(keys))[value].reindex(idx)
    if s.isna().any():
        raise ValueError(f"missing rows for {value}")
    return s.to_numpy(dtype=float).reshape(NS, NP, NT)


def point_forecast(history, column):
    """Section 2.3 rule: last-7-day level x store weekday index (28 days), on `column`."""
    h = history.copy()
    h["dt"] = pd.to_datetime(h["dt"])
    level = h[h.dt.between("2024-06-19", "2024-06-25")].groupby(
        ["store_id", "product_id"])[column].mean()
    tot = h[h.dt.between("2024-05-29", "2024-06-25")].groupby(
        ["store_id", "dt"], as_index=False)[column].sum()
    tot["wd"] = tot.dt.dt.dayofweek
    g = tot.groupby(["store_id", "wd"])[column].mean() / tot.groupby("store_id")[column].mean()
    out = np.zeros((NS, NP, NT))
    for i, s in enumerate(STORES):
        for j, p in enumerate(PRODUCTS):
            for k, d in enumerate(DATES):
                out[i, j, k] = level.loc[(s, p)] * g.loc[(s, pd.Timestamp(d).dayofweek)]
    return out


def ranking_totals(history, column):
    """Total input demand per store-product over May 29-June 25 (selection rule)."""
    h = history[(history.dt >= "2024-05-29") & (history.dt <= "2024-06-25")]
    tot = h.groupby(["store_id", "product_id"])[column].sum()
    return np.array([[tot.loc[(s, p)] for p in PRODUCTS] for s in STORES])


# ----------------------------------------------------------------------------- simulator
def simulate(q, d):
    """FIFO two-day-life simulation of a fixed plan q[s,p,t] against demand d[s,p,t].

    Returns a dict of arrays with the same shape: O, V, A, Sold, Snew, L, W, U.
    """
    q = np.asarray(q, float)
    d = np.asarray(d, float)
    out = {k: np.zeros_like(d) for k in ("O", "V", "A", "Sold", "Snew", "L", "W", "U")}
    old = np.zeros(d.shape[:2])
    for t in range(d.shape[2]):
        new, dem = q[:, :, t], d[:, :, t]
        avail = old + new
        sold_old = np.minimum(old, dem)
        sales = np.minimum(avail, dem)
        sold_new = sales - sold_old
        left = new - sold_new
        for k, v in (("O", old), ("V", avail), ("A", sales), ("Sold", sold_old),
                     ("Snew", sold_new), ("L", left), ("W", old - sold_old), ("U", dem - sales)):
            out[k][:, :, t] = v
        old = left
    return out


def metrics(q, z, d, prm=BASE):
    """Operational metrics and weighted loss J for one plan and one demand path."""
    o = simulate(q, d)
    D, A, U, W = d.sum(), o["A"].sum(), o["U"].sum(), o["W"].sum()
    carry, term = o["L"][:, :, :-1].sum(), o["L"][:, :, -1].sum()
    unselected = (d * (1 - np.asarray(z))[:, :, None]).sum()
    loss = prm["alpha"] * U + prm["beta"] * W + prm["h"] * carry + prm["tau"] * term
    return dict(demand=D, sales=A, fill_rate=A / D, unmet=U, unmet_unselected=unselected,
                unmet_selected=U - unselected, expired=W, carryover=carry, day7_stock=term,
                delivered=float(np.sum(q)), loss=loss)


def check_feasible(q, z, scen, prm=BASE, tol=1e-7):
    """Assortment, delivery link, regional supply, and store capacity in every scenario."""
    q, z = np.asarray(q), np.asarray(z)
    worst = dict(
        assortment=max(0, (z.sum(axis=1) - prm["K"]).max()),
        delivery_link=max(0, (q - 1e6 * z[:, :, None]).max()),
        negative=max(0, -q.min()),
        supply=max(0, (q.sum(axis=(0, 1)) - prm["R"]).max()))
    cap = np.array([prm["C"][s] for s in STORES])[:, None]
    worst["capacity"] = max(0, max((simulate(q, d)["V"].sum(axis=1) - cap).max()
                                   for d in scen.values()))
    if max(worst.values()) > tol:
        raise RuntimeError(f"infeasible plan: {worst}")
    return worst


# ----------------------------------------------------------------------------- baseline
def baseline_plan(totals, fc, scen, prm=BASE):
    """Rule-based baseline.

    1. Each store carries its K products with the highest May 29-June 25 input demand
       (ties broken by product ID).
    2. Each carried product orders its point forecast; no carryover adjustment.
    3. Days whose regional total exceeds R are scaled down proportionally.
    4. Day by day, each store's day-t quantities are scaled by
       lambda = min(1, min_w (C_s - sum_p O_spt,w) / sum_p q_spt), so the plan fits store
       capacity in every planning scenario. Old stock on day t depends only on earlier days,
       so one forward pass suffices.
    """
    z = np.zeros((NS, NP))
    for i in range(NS):
        order = sorted(range(NP), key=lambda j: (-totals[i, j], PRODUCTS[j]))
        z[i, order[:prm["K"]]] = 1
    q = fc * z[:, :, None]
    day = q.sum(axis=(0, 1))
    q = q * np.minimum(1.0, prm["R"] / np.where(day > 0, day, 1))[None, None, :]
    for t in range(NT):
        for i, s in enumerate(STORES):
            tot_q = q[i, :, t].sum()
            if tot_q <= 0:
                continue
            lam = 1.0
            for d in scen.values():
                old_sum = simulate(q[:, :, :t + 1], d[:, :, :t + 1])["O"][i, :, t].sum()
                lam = min(lam, (prm["C"][s] - old_sum) / tot_q)
            q[i, :, t] *= max(0.0, lam)
    return q, z


# ----------------------------------------------------------------------------- MILP (HiGHS)
def solve_milp(scen, pi, prm=BASE, time_limit=300, gap=1e-3, threads=2, log=None, start=None):
    """The Milestone 4 MILP, built for HiGHS (used where Gurobi is not available).

    Same variables, constraints (2)-(11) and objective (1) as the Milestone 4 notebook.
    start: optional feasible plan (q, z) used as a MIP start; all other variables are
    filled in by simulating that plan in every scenario.
    """
    import highspy
    from scipy.sparse import coo_matrix  # noqa: F401  (only to fail early if missing)

    S, P = range(NS), range(NP)
    col_lb, col_ub, col_cost, integ, sv = [], [], [], [], []
    q0, z0 = (np.asarray(start[0], float), np.asarray(start[1], float)) if start else (None, None)

    def add(lb, ub, cost=0.0, binary=False, val=0.0):
        col_lb.append(lb); col_ub.append(ub); col_cost.append(cost); integ.append(binary)
        sv.append(float(val))
        return len(col_lb) - 1

    C = {i: prm["C"][s] for i, s in enumerate(STORES)}
    z = {(i, j): add(0, 1, binary=True, val=z0[i, j] if start else 0)
         for i in S for j in P}
    q = {(i, j, t): add(0, min(C[i], prm["R"]), val=q0[i, j, t] if start else 0)
         for i in S for j in P for t in range(NT)}
    V = {}
    rows = []  # (dict col->coef, lo, hi)

    def row(coefs, lo, hi):
        rows.append((coefs, lo, hi))

    inf = highspy.kHighsInf
    for i in S:
        row({z[i, j]: 1 for j in P}, -inf, prm["K"])
    for i, j, t in q:
        row({q[i, j, t]: 1, z[i, j]: -min(C[i], prm["R"])}, -inf, 0)
    for t in range(NT):
        row({q[i, j, t]: 1 for i in S for j in P}, -inf, prm["R"])
    for w in scen:  # any number of scenarios (3 planning scenarios or sampled paths)
        d = scen[w]
        p_w = pi[w]
        L_prev = {}
        o = simulate(q0, d) if start else None

        def sv_(key, i, j, t):
            return o[key][i, j, t] if start else 0.0
        for t in range(NT):
            for i in S:
                for j in P:
                    dem = d[i, j, t]
                    B = max(C[i], dem)
                    O = add(0, C[i], val=sv_("O", i, j, t))
                    Vv = add(0, C[i], val=sv_("V", i, j, t))
                    A = add(0, min(C[i], dem), val=sv_("A", i, j, t))
                    So = add(0, min(C[i], dem), val=sv_("Sold", i, j, t))
                    Sn = add(0, C[i], val=sv_("Snew", i, j, t))
                    is_last = t == NT - 1
                    L = add(0, C[i], p_w * (prm["tau"] if is_last else prm["h"]),
                            val=sv_("L", i, j, t))
                    W = add(0, C[i], p_w * prm["beta"], val=sv_("W", i, j, t))
                    U = add(0, dem, p_w * prm["alpha"], val=sv_("U", i, j, t))
                    bA = add(0, 1, binary=True,
                             val=float(start is not None and sv_("V", i, j, t) > dem))
                    bO = add(0, 1, binary=True,
                             val=float(start is not None and sv_("O", i, j, t) > dem))
                    V[i, j, t, w] = Vv
                    if t == 0:
                        row({O: 1}, 0, 0)
                    else:
                        row({O: 1, L_prev[i, j]: -1}, 0, 0)
                    row({Vv: 1, O: -1, q[i, j, t]: -1}, 0, 0)
                    row({A: 1, Vv: -1}, -inf, 0)
                    row({A: 1}, -inf, dem)
                    row({A: 1, Vv: -1, bA: B}, 0, inf)
                    row({A: 1, bA: -B}, dem - B, inf)
                    row({So: 1, O: -1}, -inf, 0)
                    row({So: 1}, -inf, dem)
                    row({So: 1, O: -1, bO: B}, 0, inf)
                    row({So: 1, bO: -B}, dem - B, inf)
                    row({Sn: 1, A: -1, So: 1}, 0, 0)
                    row({L: 1, q[i, j, t]: -1, Sn: 1}, 0, 0)
                    row({W: 1, O: -1, So: 1}, 0, 0)
                    row({U: 1, A: 1}, dem, dem)
                    L_prev[i, j] = L
            for i in S:
                row({V[i, j, t, w]: 1 for j in P}, -inf, C[i])

    lp = highspy.HighsLp()
    n = len(col_lb)
    lp.num_col_, lp.num_row_ = n, len(rows)
    lp.col_cost_ = np.array(col_cost)
    lp.col_lower_, lp.col_upper_ = np.array(col_lb, float), np.array(col_ub, float)
    lp.row_lower_ = np.array([r[1] for r in rows], float)
    lp.row_upper_ = np.array([r[2] for r in rows], float)
    starts, idx, val = [0], [], []
    for coefs, _, _ in rows:
        idx.extend(coefs.keys()); val.extend(coefs.values()); starts.append(len(idx))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = starts, idx, val
    lp.integrality_ = [highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous
                       for b in integ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", log is not None)
    if log:
        h.setOptionValue("log_file", str(log))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("threads", threads)
    h.setOptionValue("random_seed", 21)
    h.passModel(lp)
    if start:
        sol = highspy.HighsSolution()
        sol.col_value = sv
        sol.value_valid = True
        h.setSolution(sol)
    h.run()
    info = h.getInfo()
    x = np.array(h.getSolution().col_value)
    qa = np.array([[[x[q[i, j, t]] for t in range(NT)] for j in P] for i in S])
    za = np.array([[round(x[z[i, j]]) for j in P] for i in S])
    qa[qa < 1e-9] = 0.0
    status = h.modelStatusToString(h.getModelStatus())
    return qa, za, dict(status=status, objective=info.objective_function_value,
                        bound=info.mip_dual_bound, gap=info.mip_gap,
                        vars=n, rows=len(rows))


# ----------------------------------------------------------------------------- io helpers
def plan_from_repo(results_dir):
    dl = pd.read_csv(Path(results_dir) / "deliveries.csv")
    asg = pd.read_csv(Path(results_dir) / "assortment.csv")
    q = to_array(dl, "delivery")
    z = asg.set_index(["store_id", "product_id"]).selected.reindex(
        pd.MultiIndex.from_product([STORES, PRODUCTS])).to_numpy().reshape(NS, NP)
    return q, z.astype(float)


def plan_to_long(q, z, policy):
    rows = []
    for (i, s), (j, p), (k, dt) in product(enumerate(STORES), enumerate(PRODUCTS), enumerate(DATES)):
        rows.append(dict(policy=policy, store_id=s, product_id=p, dt=dt,
                         selected=int(z[i, j]), delivery=q[i, j, k]))
    return pd.DataFrame(rows)
