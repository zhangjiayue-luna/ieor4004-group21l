"""Group 21L, Milestone 5: score every plan on the Monte Carlo demand paths.

Usage:
    python mc_evaluation.py --plans <run dir>/plans_long.csv --paths mc_paths.csv --out <dir>

Each path is one full week of product-level demand (3 stores x 29 products x 7 days).
Every plan is scored on the same paths with the same simulator, so plan comparisons are
paired path by path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as ev  # noqa: E402


def load_paths(path):
    mp = pd.read_csv(path)
    cols = [f"s{s}_p{p}_t{t}" for s in ev.STORES for p in ev.PRODUCTS for t in ev.T]
    if list(mp.columns[1:]) != cols or mp[cols].isna().any().any() or (mp[cols] < 0).any().any():
        raise ValueError("unexpected Monte Carlo path layout or values")
    return mp.path_id.to_numpy(), mp[cols].to_numpy(float).reshape(-1, ev.NS, ev.NP, ev.NT)


def score_paths(q, z, X, prm=ev.BASE):
    """Vectorized simulation of one plan over N paths. Returns a per-path DataFrame."""
    n = X.shape[0]
    d = X.reshape(n * ev.NS, ev.NP, ev.NT)
    o = ev.simulate(np.tile(q, (n, 1, 1)), d)
    shp = (n, ev.NS, ev.NP, ev.NT)
    U, W, L, V = (o[k].reshape(shp) for k in ("U", "W", "L", "V"))
    A = o["A"].reshape(shp)
    D = X.sum(axis=(1, 2, 3))
    carry, term = L[..., :-1].sum(axis=(1, 2, 3)), L[..., -1].sum(axis=(1, 2))
    loss = (prm["alpha"] * U.sum(axis=(1, 2, 3)) + prm["beta"] * W.sum(axis=(1, 2, 3))
            + prm["h"] * carry + prm["tau"] * term)
    cap = np.array([prm["C"][s] for s in ev.STORES])[None, :, None]
    over = V.sum(axis=2) - cap
    unselected = (X * (1 - z)[None, :, :, None]).sum(axis=(1, 2, 3))
    return pd.DataFrame(dict(demand=D, sales=A.sum(axis=(1, 2, 3)), fill_rate=A.sum(axis=(1, 2, 3)) / D,
                             unmet=U.sum(axis=(1, 2, 3)), unmet_unselected=unselected,
                             expired=W.sum(axis=(1, 2, 3)), carryover=carry, day7_stock=term,
                             loss=loss, cap_days_over=(over > 1e-6).sum(axis=(1, 2)),
                             cap_max_over=np.maximum(over.max(axis=(1, 2)), 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids, X = load_paths(args.paths)
    plans = pd.read_csv(args.plans)
    per_path = []
    for pol, sub in plans.groupby("policy", sort=False):
        q = ev.to_array(sub, "delivery")
        z = (sub.groupby(["store_id", "product_id"]).selected.first()
             .reindex(pd.MultiIndex.from_product([ev.STORES, ev.PRODUCTS])).to_numpy()
             .reshape(ev.NS, ev.NP).astype(float))
        df = score_paths(q, z, X)
        df.insert(0, "path_id", ids)
        df.insert(0, "policy", pol)
        per_path.append(df)
    per_path = pd.concat(per_path, ignore_index=True)
    per_path.to_csv(out / "mc_per_path.csv", index=False)

    # summary: mean with 95% CI, plus loss percentiles
    rows = []
    for pol, g in per_path.groupby("policy", sort=False):
        r = dict(policy=pol, n_paths=len(g))
        for c in ("fill_rate", "unmet", "unmet_unselected", "expired", "carryover", "day7_stock", "loss"):
            r[c] = g[c].mean()
        se = g.loss.std(ddof=1) / np.sqrt(len(g))
        r.update(loss_ci_low=r["loss"] - 1.96 * se, loss_ci_high=r["loss"] + 1.96 * se,
                 loss_p5=g.loss.quantile(0.05), loss_p50=g.loss.median(), loss_p95=g.loss.quantile(0.95),
                 share_paths_cap_exceeded=(g.cap_days_over > 0).mean(),
                 mean_cap_days_over=g.cap_days_over.mean(), mean_cap_max_over=g.cap_max_over.mean())
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "mc_summary.csv", index=False)

    # paired differences (same path, two plans)
    wide = per_path.pivot(index="path_id", columns="policy", values="loss")
    pairs = [("M-rec", "B-rec"), ("M-rec-Gurobi", "B-rec"), ("M-raw", "B-raw"),
             ("B-rec", "B-raw"), ("M-rec", "M-raw")]
    if "M-mc" in wide:
        pairs += [("M-mc", "B-mc"), ("M-mc", "M-rec")]
    pairs += [(c, "B-rec") for c in wide.columns if c.startswith("M-saa")]
    prow = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        diff = wide[a] - wide[b]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        prow.append(dict(comparison=f"{a} minus {b}", mean_diff=diff.mean(),
                         ci_low=diff.mean() - 1.96 * se, ci_high=diff.mean() + 1.96 * se,
                         pct_of_second=diff.mean() / wide[b].mean() * 100,
                         share_paths_first_better=(diff < 0).mean()))
    paired = pd.DataFrame(prow)
    paired.to_csv(out / "mc_paired_differences.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(paired.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
