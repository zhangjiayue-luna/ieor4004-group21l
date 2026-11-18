"""Reproduce MC policy scoring on the archived corrected paths and saved plans."""
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "evaluation/reproduced_mc"
    output.mkdir(parents=True, exist_ok=True)
    paths = pd.read_csv(ROOT / "data/mc_paths.csv")
    scoring = paths.loc[~paths.path_id.isin(range(1, 11))]
    if len(scoring) != 1990 or scoring.path_id.nunique() != 1990:
        raise ValueError("Expected 1,990 distinct scoring paths after reserving 1–10")
    filename = output / "scoring_paths.csv"
    scoring.to_csv(filename, index=False)
    command = [sys.executable, str(ROOT / "evaluation/code/mc_evaluation.py"),
               "--plans", str(ROOT / "evaluation/results_mc/plans_all.csv"),
               "--paths", str(filename), "--out", str(output)]
    subprocess.run(command, cwd=ROOT, check=True)
    actual = pd.read_csv(output / "mc_summary.csv").set_index("policy")
    reference = pd.read_csv(ROOT / "evaluation/results_mc/mc_summary.csv").set_index("policy")
    for col in ["loss", "fill_rate", "expired", "unmet"]:
        error = (actual[col] - reference[col]).abs().max()
        if error > 1e-7:
            raise ValueError(f"Archived MC result differs: {col}, error {error}")
    per_path = pd.read_csv(output / "mc_per_path.csv")
    stats = per_path.groupby("policy").loss.agg(["count", "mean", "std"])
    stats.to_csv(output / "loss_mean_sd.csv")
    print(stats.round(4).to_string())
    print("Verified scoring matches the archived v3 results.")


if __name__ == "__main__":
    main()
