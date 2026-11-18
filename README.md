# IEOR E4004 — Group 21L — Milestone 5

Jiayue Zhang, Yue Peng, Andrew Bian, Langzhuo Liu, Hang Seo Cho

## Results and files

The report contains the complete MILP, original optimization outcomes, 14 sensitivity
cases, cutoff-corrected demand simulation, holdout comparisons and common-path MC
evaluation. Quantities retain the normalized dataset scale. Loss weights are relative
preferences. Out-of-sample evaluation records overflow while permitting stock to
continue through the simulator. Recommendations acknowledge that operating convention.

- `gurobi_model.py`: original M4 model, inventory reconstruction, and verification.
- `sensitivity_analysis.py`: Yue's parameter experiments and conditional LP duals.
- `optimization_analysis.py`: recorded-result audit and optimization/sensitivity charts.
- `Group21L_M5_Optimization_Analysis.ipynb`: executed report tables and figures.
- `data/`: original planning inputs, holdout recovery inputs, corrected `mc_paths.csv`
  (2,000 paths) and `mc_scenarios_1827.csv`.
- `results/`: M4 and sensitivity outputs, model/solution records, and report audits.
- `evaluation/code/`: Hang's v3 common inventory simulator, HiGHS model, policy scoring,
  and figure-generation scripts.
- `evaluation/results/`: holdout and planning-scenario four-way comparisons.
- `evaluation/results_mc/`: saved plans and v3 MC scores on paths 11–2000.
- `evaluation/mc_method.tex`: supplied description of the corrected simulation method.
- `report/`: final LaTeX source, figures, and report PDF.

## Environment and recorded-result reproduction

Install with `python -m pip install -r requirements.txt`. A valid personal Gurobi
license is needed for new Gurobi solves. License credentials are excluded.

From the repository root:

```bash
python optimization_analysis.py
python reproduce_m5.py
```

The first command reconciles recorded scenario balances and losses and creates
optimization charts. The second reconstructs every saved policy's MC score from
the corrected paths, excludes diagnostic paths 1–10, verifies reference metrics,
and exports sample SDs. It uses the archived plans and requires no optimization solve.

Optional new solves, which can change time-limited incumbents:

```bash
python optimization_analysis.py --solve-base
python optimization_analysis.py --solve-sensitivity --time-limit 60
python evaluation/code/run_evaluation.py --repo . --eval data --out evaluation/new_results --time-limit 900
python evaluation/code/extra_plans.py --repo . --mc-scen data/mc_scenarios_1827.csv --mc-paths data/mc_paths.csv --out evaluation/new_mc_plans --time-limit 900
```

Keep planning paths and scoring paths separate when evaluating a new plan. Seed,
time limits, solver versions, achieved bounds and gaps are retained in result files.
MC fill rates average path-level ratios. Sensitivity fill rate is expected sales
divided by expected demand. All plan differences in MC use common paired paths.

## Simulation provenance and limitations

The archived corrected paths and scenario CSV were supplied by the demand team.
All eight backtest forecasts use their cutoff-specific recovery files, as described
in `evaluation/mc_method.tex`. Realized demand remains estimated from full-period
recovered demand. The supplied materials do not include the simulation-generation
script or all eight cutoff recovery files. This package reproduces scoring on the
provided simulation paths; independent regeneration of those paths requires these
additional upstream materials. Reported leave-one-week-out validation is attributed
to the demand team's method document.

The ten-path solver lower bound concerns a stricter feasible set than the original
overflow-permitting baseline. The diagnostic incumbent matches its repaired
baseline start and has a 13.7% gap. Interpret this as a feasibility/service diagnostic.

AI assisted implementation, verification, graphics and report wording. Team members
retain responsibility for submitted code, assumptions and interpretation.
