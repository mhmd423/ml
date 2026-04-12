# ML Learning Repo

This repository is a small machine learning practice space where classic models are implemented from scratch and tried on course datasets.

## What's here

- `src/models/`: in-repo Python package containing the base `Model` class and separate modules for `LogisticRegression`, `LinearRegression`, `GDA`, and `PoissonRegression`.
- `src/metrics/`: reusable evaluation helpers such as classification accuracy.
- `src/utils.py`: shared data-loading utility functions.
- `scripts/ps1.py`: a CS229-style logistic regression example.
- `notebooks/`: notebook experiments such as `lwlr.ipynb` and `ps1.ipynb`.
- `data/`: local datasets (`data/cs229-2018-autumn/`, `data/datasets/`).

## How to run

Create or activate a Python environment, then install the package:

```bash
pip install -e .
```

For development (tests), install dev extras:

```bash
pip install -e ".[dev]"
```

Then run the PS1 example:

```bash
python scripts/ps1.py
```

To run the test suite:

```bash
python -m pytest -q
```

This will:

- load the PS1 training and validation datasets
- fit logistic regression with Newton's method
- print training and validation accuracy
- display the training decision boundary and loss curve

## Notes

- The CS229 datasets live under `data/cs229-2018-autumn/`, which is ignored in git because it is treated as external course material.
- Most of the work in this repo is educational and exploratory, so notebooks and scripts are both present.
