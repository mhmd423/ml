# ML Learning Repo

This repository is a small machine learning practice space where classic models are implemented from scratch and tried on course datasets.

## What's here

- `models.py`: base `Model` class plus custom `LogisticRegression` and `LinearRegression` implementations using NumPy.
- `metrics.py`: reusable evaluation helpers such as classification accuracy.
- `ps1.py`: a CS229-style logistic regression example that trains on `ds1_train.csv` and evaluates on `ds1_valid.csv`.
- `lwlr.ipynb` and `ps1.ipynb`: notebook experiments.
- `da.py`: helper script for downloading the housing dataset used in other ML exercises.

## How to run

Create or activate a Python environment with the packages used in the repo:

```bash
pip install numpy pandas matplotlib
```

Then run:

```bash
python ps1.py
```

This will:

- load the PS1 training and validation datasets
- fit logistic regression with Newton's method
- print training and validation accuracy
- display the training decision boundary and loss curve

## Notes

- The CS229 datasets live under `cs229-2018-autumn/`, which is ignored in git because it is treated as external course material.
- Most of the work in this repo is educational and exploratory, so notebooks and scripts are both present.
