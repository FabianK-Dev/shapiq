"""
OddSHAP: interaction-sparsity (eta) ablation
============================================

Reproduces the paper's Figure 4 (Fumagalli et al. 2026, arXiv:2602.01399,
Section 5.3): the impact of the number of selected odd interactions
``|T_odd| = ceil(m / eta)`` on accuracy, at the paper's **fixed budget of
10,000 samples**, for ``eta`` in {50, 10, 5, 2}. The interaction-free baseline
(OddSHAP with zero higher-order odd interactions — the paper's
LeverageSHAP-equivalent) provides the normalisation, so the reported quantity
is the paper's *MSE ratio* (< 1 = the screened interactions help; the ratio
exploding at ``eta = 2`` is the paper's overfitting regime).

All tabular value functions use XGBoost classifiers (paper Section 5) and the
exact interventional Shapley ground truth from shapiq's own tree explainer.
``N_INSTANCES = 30`` reproduces the report's numbers; the default below is
reduced so the gallery builds quickly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from shapiq.approximator import OddSHAP
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq_games import datasets
from shapiq_games.benchmark.interventionaltreeshapiq_xai import InterventionalGame

RANDOM_STATE = 40
N_INSTANCES = 3  # the report uses 30; reduced here for a fast gallery build
N_BACKGROUND = 50
BUDGET = 10_000  # paper Section 5.3: fixed budget of 10,000 samples
ETAS = [50, 10, 5, 2]  # |T_odd| = 200 / 1,000 / 2,000 / 5,000

# (name, loader, binarize) — all classifiers per paper Section 5
VALUE_FUNCTIONS = [
    ("Cancer", datasets.load_breast_cancer, False),
    ("Estate", datasets.load_real_estate, True),
    ("CG60", datasets.load_corrgroups60, True),
    ("IL60", datasets.load_independentlinear60, True),
    ("NHANES", datasets.load_nhanesi, True),
    ("Crime", datasets.load_communities_and_crime, True),
]


def single_feature_values(interaction_values, n: int) -> np.ndarray:
    return np.array([float(interaction_values.dict_values.get((i,), 0.0)) for i in range(n)])


ratios: dict[str, list[float]] = {}
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

    per_eta: dict[int | str, list[float]] = {eta: [] for eta in ETAS}
    per_eta["base"] = []
    for i in range(min(N_INSTANCES, x_te.shape[0])):
        target = x_te[i]
        gt = single_feature_values(ground_truth.explain_function(target.astype(np.float32)), n)
        game = InterventionalGame(model=model, reference_data=bg, target_instance=target,
                                  class_index=1)
        for eta in ETAS:
            estimator = OddSHAP(n=n, random_state=0, interaction_factor=eta)
            iv = estimator.approximate(BUDGET, game)
            per_eta[eta].append(float(np.mean((single_feature_values(iv, n) - gt) ** 2)))
        # interaction-free baseline: OddSHAP with an empty higher-order odd support
        estimator = OddSHAP(n=n, random_state=0, interaction_factor=10)
        estimator._select_odd_interactions = lambda **kw: []
        iv = estimator.approximate(BUDGET, game)
        per_eta["base"].append(float(np.mean((single_feature_values(iv, n) - gt) ** 2)))

    base = float(np.median(per_eta["base"]))
    ratios[vf_name] = [float(np.median(per_eta[eta])) / base for eta in ETAS]
    pretty = ", ".join(f"eta={eta}: {r:.3f}" for eta, r in zip(ETAS, ratios[vf_name]))
    print(f"{vf_name:8s} (d={n:3d})  MSE ratio vs interaction-free baseline: {pretty}")

# %%
# Plot — the paper's Figure-4 shape: improvement for moderate |T_odd|,
# overfitting blow-up for eta = 2.
n_interactions = [int(np.ceil(BUDGET / eta)) for eta in ETAS]
fig, ax = plt.subplots(figsize=(7, 4))
for vf_name, r in ratios.items():
    ax.plot(n_interactions, r, marker="o", label=vf_name)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"number of odd interactions $\lceil m/\eta \rceil$")
ax.set_ylabel("MSE ratio (vs interaction-free baseline)")
ax.set_title(f"OddSHAP interaction-sparsity ablation (budget = {BUDGET:,})")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
