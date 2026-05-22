"""Tabular value functions for the OddSHAP paper Table-1 reproduction.

The OddSHAP paper (arXiv:2602.01399) evaluates its tabular value functions
as path-dependent TreeSHAP games over a RandomForest — exactly the setup of
``experiments/approximation_pathdependent.py`` in the PolySHAP / OddSHAP
authors' code: a forest is trained, each local explanation is a
``TreeSHAPIQXAI`` game, and the exact ground truth is that game's built-in
``exact_values`` (polynomial TreeSHAP-IQ, feasible up to d=101).

This module reproduces that path. It provides, for each of the paper's
tabular value functions, a list of ``TreeSHAPIQXAI`` game instances:

    realestate           d = 15   Estate    (UCI real-estate valuation)
    cancer               d = 30   Cancer    (Wisconsin breast cancer)
    independentlinear60  d = 60   IL60      (shap synthetic)
    corrgroups60         d = 60   CG60      (shap synthetic)
    nhanes               d = 79   NHANES    (shap NHANES I)
    crime                d = 101  Crime     (shap communities-and-crime)

Dataset loaders mirror the authors' ``experiments/custom_datasets.py``.
``cancer`` uses shapiq's packaged ``BreastCancer`` game so its RandomForest
is the paper's; the rest train a ``RandomForestRegressor`` here, so their
absolute MSE is harness-relative — the paper's *conclusion* (OddSHAP ranks
best) is what those columns verify, not the absolute Table-1 cell.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# RandomForest kept modest: the path-dependent value function recurses both
# ways through every absent-feature split, so shallow trees keep the
# approximators' game calls affordable for the d=60-101 value functions.
_RF_KWARGS = {"n_estimators": 10, "max_depth": 6, "random_state": 40}
_RANDOM_STATE = 40


# -----------------------------------------------------------------------------
# Dataset loaders — mirror experiments/custom_datasets.py of the authors' repo
# -----------------------------------------------------------------------------


def _load_real_estate():
    import pandas as pd

    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00477/Real%20estate%20valuation%20data%20set.xlsx")
    df = pd.read_excel(url)
    df = df.drop(columns=["No"])
    df["month"] = (df["X1 transaction date"] % 1 * 12).round().astype(int)
    df["month"] = df["month"].replace({0: 1, 12: 1})
    df = df.drop(columns=["X1 transaction date"])
    df = pd.get_dummies(df, columns=["month"], drop_first=True)
    y = df["Y house price of unit area"].astype(float)
    x = df.drop(columns=["Y house price of unit area"])
    return x, y


def _load_corrgroups60():
    import shap

    return shap.datasets.corrgroups60()


def _load_independentlinear60():
    import shap

    return shap.datasets.independentlinear60()


def _load_nhanes():
    import shap

    return shap.datasets.nhanesi()


def _load_crime():
    import shap

    return shap.datasets.communitiesandcrime()


# name -> (loader, expected d). ``cancer`` is handled separately.
TABULAR_LOADERS = {
    "realestate": (_load_real_estate, 15),
    "independentlinear60": (_load_independentlinear60, 60),
    "corrgroups60": (_load_corrgroups60, 60),
    "nhanes": (_load_nhanes, 79),
    "crime": (_load_crime, 101),
}


# -----------------------------------------------------------------------------
# TreeSHAPIQXAI game construction
# -----------------------------------------------------------------------------


def _make_regression_tree_games(name: str, n_instances: int):
    """Train a RandomForestRegressor and yield TreeSHAPIQXAI games for it."""
    import pandas as pd  # noqa: F401  (loaders return DataFrames)

    from shapiq_games.benchmark.treeshapiq_xai import TreeSHAPIQXAI

    loader, _ = TABULAR_LOADERS[name]
    x, y = loader()
    x = x.fillna(x.mean())
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)

    x_train, x_test, y_train, _ = train_test_split(
        x_values, y_values, test_size=0.2, random_state=_RANDOM_STATE,
    )
    forest = RandomForestRegressor(**_RF_KWARGS)
    forest.fit(x_train, y_train)

    for idx in range(min(n_instances, x_test.shape[0])):
        game = TreeSHAPIQXAI(x_test[idx], forest, verbose=False)
        yield game, f"{name}_{idx}"


def _make_cancer_tree_games(n_instances: int):
    """Yield TreeSHAPIQXAI games over shapiq's packaged BreastCancer forest."""
    from shapiq_games.benchmark.local_xai import BreastCancer
    from shapiq_games.benchmark.treeshapiq_xai import TreeSHAPIQXAI

    base = BreastCancer(
        model_name="random_forest", x=0, random_state=_RANDOM_STATE,
    )
    model = base.setup.model
    x_test = np.asarray(base.setup.x_test, dtype=float)
    for idx in range(min(n_instances, x_test.shape[0])):
        game = TreeSHAPIQXAI(x_test[idx], model, verbose=False)
        yield game, f"cancer_{idx}"


def make_tabular_games(name: str, n_instances: int):
    """Yield ``(TreeSHAPIQXAI game, label)`` pairs for a paper value function.

    Args:
        name: One of ``cancer`` or a key of :data:`TABULAR_LOADERS`.
        n_instances: Number of local-explanation instances (paper uses 30).
    """
    if name == "cancer":
        yield from _make_cancer_tree_games(n_instances)
    elif name in TABULAR_LOADERS:
        yield from _make_regression_tree_games(name, n_instances)
    else:
        msg = f"Unknown tabular value function '{name}'."
        raise ValueError(msg)


TABULAR_GAMES = ["realestate", "cancer", "independentlinear60",
                 "corrgroups60", "nhanes", "crime"]
