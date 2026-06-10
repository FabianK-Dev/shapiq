"""
OddSHAP: approximation quality vs sampling budget
=================================================

Reproduces the paper's Figure 2 (Fumagalli et al. 2026, arXiv:2602.01399):
median MSE against the exact interventional Shapley values as a function of
the sampling budget ``m``, with ``m`` log-spaced from ``d + 1`` to
``min(2^d, 20000)`` (paper Section 5, "Estimators").

All tabular value functions use XGBoost classifiers (paper Section 5). Budgets
below an estimator's internal minimum (OddSHAP raises ``ValueError`` instead
of silently falling back) are skipped, so the OddSHAP curve starts at its
admissible regime — by design.

``N_INSTANCES = 10`` and the full six value functions reproduce the report's
curves; the defaults below are reduced so the gallery builds quickly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from shapiq.approximator import (
    SVARM,
    KernelSHAP,
    OddSHAP,
    PermutationSamplingSV,
    RegressionMSR,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq_games import datasets
from shapiq_games.benchmark.interventionaltreeshapiq_xai import InterventionalGame

RANDOM_STATE = 40
N_INSTANCES = 3  # the report uses 10; reduced here for a fast gallery build
N_BACKGROUND = 50
N_BUDGETS = 10
BUDGET_CAP = 20_000  # paper: m up to min(2^d, 20000)

# (name, loader, binarize) — all classifiers per paper Section 5. The full
# six-value-function list is in the report; two are kept here for build speed.
VALUE_FUNCTIONS = [
    ("Cancer", datasets.load_breast_cancer, False),
    ("NHANES", datasets.load_nhanesi, True),
]

ESTIMATORS = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]


def make_estimator(name: str, n: int):
    if name == "MSR":
        return UnbiasedKernelSHAP(n=n, index="SV", max_order=1, random_state=0)
    if name == "SVARM":
        return SVARM(n=n, random_state=0)
    if name == "PermSamp":
        return PermutationSamplingSV(n=n, random_state=0)
    if name == "KernelSHAP":
        return KernelSHAP(n=n, random_state=0)
    if name == "kADDSHAP":
        return kADDSHAP(n=n, max_order=2, random_state=0)
    if name == "RegressionMSR":
        return RegressionMSR(n=n, index="SV", random_state=0)
    return OddSHAP(n=n, random_state=0)


def single_feature_values(interaction_values, n: int) -> np.ndarray:
    return np.array([float(interaction_values.dict_values.get((i,), 0.0)) for i in range(n)])


for vf_name, loader, binarize in VALUE_FUNCTIONS:
    x, y = loader()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y = (y > np.median(y)).astype(int) if binarize else y.astype(int)
    n = x.shape[1]
    x_tr, x_te, y_tr, _ = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1).fit(x_tr, y_tr)
    rng = np.random.default_rng(RANDOM_STATE)
    bg = x_tr[rng.choice(x_tr.shape[0], size=min(N_BACKGROUND, x_tr.shape[0]), replace=False)]
    ground_truth = InterventionalTreeExplainer(
        model=model, data=bg.astype(np.float32), index="SV", max_order=1, class_index=1,
    )

    hi = min(2 ** n, BUDGET_CAP)
    budgets = sorted({int(round(b)) for b in np.logspace(np.log10(n + 1), np.log10(hi), N_BUDGETS)})
    errors: dict[str, dict[int, list[float]]] = {est: {b: [] for b in budgets} for est in ESTIMATORS}

    for i in range(min(N_INSTANCES, x_te.shape[0])):
        target = x_te[i]
        gt = single_feature_values(ground_truth.explain_function(target.astype(np.float32)), n)
        game = InterventionalGame(model=model, reference_data=bg, target_instance=target,
                                  class_index=1)
        for budget in budgets:
            for est in ESTIMATORS:
                try:
                    iv = make_estimator(est, n).approximate(budget, game)
                except (ValueError, RuntimeError):
                    continue  # estimator refuses this budget regime (by design)
                errors[est][budget].append(
                    float(np.mean((single_feature_values(iv, n) - gt) ** 2))
                )
    print(f"{vf_name:8s} (d={n:3d}) done — budgets {budgets[0]}..{budgets[-1]}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for est in ESTIMATORS:
        xs = [b for b in budgets if errors[est][b]]
        ys = [float(np.median(errors[est][b])) for b in xs]
        style = dict(lw=2.4, color="#CC3311") if est == "OddSHAP" else dict(lw=1.2)
        ax.plot(xs, ys, marker="o", ms=3, label=est, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sampling budget $m$")
    ax.set_ylabel("MSE (median)")
    ax.set_title(f"{vf_name} (d = {n}) — MSE vs budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
plt.show()
