# IEOR E4004 — Group 21L — Milestone 4

## Fresh Product Assortment and Stocking Allocation Across Three Dingdong Fulfillment Stores

Jiayue Zhang, Yue Peng, Andrew Bian, Langzhuo Liu, and Hang Seo Cho  
November 4, 2026

The project selects weekly product assortments and daily deliveries for three
Dingdong fulfillment stores over June 26–July 2, 2024. A mixed-integer linear
program minimizes expected relative operational loss across three demand
scenarios, with two-day sellable life, FIFO, store capacity, and shared supply.

## Repository contents

- `Group21L_Milestone4.ipynb`: model, documented assumptions, executed results,
  and verification. The notebook includes the implementation and does not
  require importing the separate Python module.
- `gurobi_model.py`: the same model and verification functions as a Python module.
- `data/`: point forecasts, scenario demand, and historical recovered demand.
- `results/base_case/`: decision tables, scenario outcomes, parameter records,
  input hashes, solver summary and log, LP model, and incumbent solution.
- `requirements.txt`: Python dependencies.

## Computational environment and execution

The reported computation used Python 3.12 and Gurobi 13.0.3. A valid Gurobi
license is required; each user supplies their own configuration. License
credentials are not included in this repository.

From the repository directory, install dependencies with
`python -m pip install -r requirements.txt`. Open `Group21L_Milestone4.ipynb`
in Jupyter with that Python environment and execute the cells in order. The
`data` folder must remain beside the notebook. The time limit is 300 seconds
and the requested relative MIP gap is 0.001. Hardware and solver versions may
affect the incumbent obtained within a fixed time limit.

## Reported computation

The incumbent objective is 1,229.3058, the best bound is 1,212.6949, and the
relative gap is 1.3512%. The run terminated at the time limit. All constraint,
inventory-flow, and objective-reconciliation checks passed. These figures
describe a feasible incumbent, not a proof of an exact global optimum.
Quantities are normalized and penalties represent relative preferences.

Data source: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K).
Software: [Gurobi Python API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html).

AI tools assisted implementation and documentation. Verification procedures
are explicitly included in the code and notebook.
