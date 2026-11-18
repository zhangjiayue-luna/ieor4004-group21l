"""Group 21L, Milestone 5: extra plans built from Langzhuo's Monte Carlo output.

  B-mc / M-mc : baseline and MILP planned on mc_scenarios_1827.csv (recalibrated
                common multipliers, same three-scenario structure)
  M-saa10     : diagnostic only - MILP planned on 10 Monte Carlo paths (equal weights),
                i.e. a small sample-average approximation that sees product-level noise.
                Paths used for planning are excluded when it is scored.

Usage: python extra_plans.py --repo <checkout> --mc-scen mc_scenarios_1827.csv
                             --mc-paths mc_paths.csv --out <dir> [--time-limit 900]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as ev  # noqa: E402
import mc_evaluation as mc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--mc-scen", required=True)
ap.add_argument("--mc-paths", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--time-limit", type=float, default=900)
ap.add_argument("--saa-n", type=int, default=10)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

hist = pd.read_csv(Path(a.repo) / "data/recovered_demand.csv")
ms = pd.read_csv(a.mc_scen)
scen = {w: ev.to_array(ms[ms.scenario == w], "demand") for w in ev.OMEGA}
pi = ms.groupby("scenario").weight.first().to_dict()
fc, tot = ev.point_forecast(hist, "demand"), ev.ranking_totals(hist, "demand")

rows, info = [], {}
q, z = ev.baseline_plan(tot, fc, scen)
ev.check_feasible(q, z, scen)
rows.append(ev.plan_to_long(q, z, "B-mc"))

q, z, inf = ev.solve_milp(scen, pi, time_limit=a.time_limit, log=out / "highs_mc.log")
ev.check_feasible(q, z, scen)
rows.append(ev.plan_to_long(q, z, "M-mc"))
info["M-mc"] = inf
pd.concat(rows).to_csv(out / "plans_extra.csv", index=False)

ids, X = mc.load_paths(a.mc_paths)
saa = {f"path{ids[k]}": X[k] for k in range(a.saa_n)}
# MIP start: the rule-based baseline repaired to fit capacity in all sampled paths
q0, z0 = ev.baseline_plan(tot, fc, saa)
ev.check_feasible(q0, z0, saa)
q, z, inf = ev.solve_milp(saa, {k: 1 / a.saa_n for k in saa}, time_limit=a.time_limit,
                          log=out / "highs_saa.log", start=(q0, z0))
rows.append(ev.plan_to_long(q0, z0, f"B-saa{a.saa_n}"))
ev.check_feasible(q, z, saa)
rows.append(ev.plan_to_long(q, z, f"M-saa{a.saa_n}"))
info[f"M-saa{a.saa_n}"] = dict(inf, planning_paths=[int(i) for i in ids[:a.saa_n]])
pd.concat(rows).to_csv(out / "plans_extra.csv", index=False)
(out / "solver_info_extra.json").write_text(json.dumps(info, indent=2, default=str))
print(json.dumps(info, indent=1, default=str))
