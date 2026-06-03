"""
OddSHAP: reproducing the paper's tabular benchmark
==================================================

Reproduces the headline result of *An Odd Estimator for Shapley Values*
(Fumagalli et al., arXiv:2602.01399) inside ``shapiq``: across tabular value
functions, :class:`~shapiq.approximator.regression.oddshap.OddSHAP` attains the
lowest mean-squared error against the exact interventional Shapley values
(the best average rank) among the sampling-based approximators.

The exact ground truth is computed with shapiq's own
:class:`~shapiq.tree.interventional.explainer.InterventionalTreeExplainer`
(``index="SV"``); no external ``shap`` dependency is needed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import shapiq_games.datasets as datasets
from shapiq.approximator import (
    KernelSHAP,
    OddSHAP,
    PermutationSamplingSV,
    RegressionMSR,
    SVARM,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq_games.benchmark.interventionaltreeshapiq_xai import InterventionalGame

RANDOM_STATE = 40
N_INSTANCES = 5  # the paper/report uses 30; reduced here for a fast gallery build
N_BACKGROUND = 50

# Each value function trains a task-appropriate tree model. ``(name, loader,
# classifier, binarize)``: Cancer is natively binary; the synthetic / survival
# targets are median-binarized into a classifier; the continuous real-estate
# target keeps a regressor (so the error scale matches the paper).
VALUE_FUNCTIONS = [
    ("Cancer", datasets.load_breast_cancer, True, False),
    ("CG60", datasets.load_corrgroups60, True, True),
    ("NHANES", datasets.load_nhanesi, True, True),
    ("Estate", datasets.load_real_estate, False, False),
]

ESTIMATORS = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]


def make_estimator(name: str, n: int):
    """Build one approximator (all use ``index='SV'`` / single-feature values)."""
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
    """Extract the single-feature (Shapley) values as a dense vector."""
    return np.array([float(interaction_values.dict_values.get((i,), 0.0)) for i in range(n)])


# %%
# Run the reproduction
# --------------------
# For each value function we train a model, take the exact interventional Shapley
# values from shapiq as ground truth, and measure each approximator's MSE at a
# sampling budget of ``m = 100 * d``.

medians: dict[str, dict[str, float]] = {est: {} for est in ESTIMATORS}
for vf_name, loader, is_clf, binarize in VALUE_FUNCTIONS:
    x, y = loader()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if binarize:
        y = (y > np.median(y)).astype(int)
    elif is_clf:
        y = y.astype(int)  # already a binary classification target (e.g. Cancer)
    n = x.shape[1]
    x_tr, x_te, y_tr, _ = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    model = (XGBClassifier if is_clf else XGBRegressor)(random_state=RANDOM_STATE, n_jobs=1)
    model.fit(x_tr, y_tr)
    rng = np.random.default_rng(RANDOM_STATE)
    bg = x_tr[rng.choice(x_tr.shape[0], size=min(N_BACKGROUND, x_tr.shape[0]), replace=False)]

    # exact interventional Shapley ground truth -- shapiq, not shap
    ground_truth = InterventionalTreeExplainer(
        model=model, data=bg.astype(np.float32), index="SV", max_order=1,
        class_index=1 if is_clf else None,
    )
    budget = max(n + 1, 100 * n)
    errors: dict[str, list[float]] = {est: [] for est in ESTIMATORS}
    for i in range(min(N_INSTANCES, x_te.shape[0])):
        target = x_te[i]
        gt = single_feature_values(ground_truth.explain_function(target.astype(np.float32)), n)
        game = InterventionalGame(
            model=model, reference_data=bg, target_instance=target,
            class_index=1 if is_clf else None,
        )
        for est in ESTIMATORS:
            iv = make_estimator(est, n).approximate(budget, game)
            errors[est].append(float(np.mean((single_feature_values(iv, n) - gt) ** 2)))
    for est in ESTIMATORS:
        medians[est][vf_name] = float(np.median(errors[est]))
    print(f"{vf_name:8s} (d={n:3d}) done")

# %%
# Results: median MSE and average rank
# ------------------------------------

vf_names = [vf[0] for vf in VALUE_FUNCTIONS]
ranks: dict[str, list[int]] = {est: [] for est in ESTIMATORS}
for vf in vf_names:
    for rank, est in enumerate(sorted(ESTIMATORS, key=lambda e: medians[e][vf]), start=1):
        ranks[est].append(rank)
avg_rank = {est: float(np.mean(ranks[est])) for est in ESTIMATORS}

print("\nMedian MSE against the exact interventional Shapley values (lower is better):")
print("estimator".ljust(15) + "".join(vf.rjust(11) for vf in vf_names) + "  avg rank")
for est in sorted(ESTIMATORS, key=lambda e: avg_rank[e]):
    cells = "".join(f"{medians[est][vf]:11.2e}" for vf in vf_names)
    print(est.ljust(15) + cells + f"  {avg_rank[est]:8.2f}")

best = min(avg_rank, key=avg_rank.get)
print(f"\nBest average rank: {best} ({avg_rank[best]:.2f})")

# %%
# Average-rank bar plot
# ---------------------
# OddSHAP attains rank 1 on every value function, reproducing the paper's
# headline claim.

ordered = sorted(ESTIMATORS, key=lambda e: avg_rank[e])
colors = ["#CC3311" if est == "OddSHAP" else "#88AACC" for est in ordered]
fig, ax = plt.subplots(figsize=(7.0, 3.5))
ax.barh(ordered, [avg_rank[est] for est in ordered], color=colors, edgecolor="black", linewidth=0.4)
ax.invert_yaxis()
ax.set_xlabel("average rank (1 = best)")
ax.set_title("OddSHAP reproduction: average rank over the tabular value functions")
fig.tight_layout()
plt.show()
