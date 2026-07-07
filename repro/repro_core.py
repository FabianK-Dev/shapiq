"""Reproduction core — one place that builds value functions, ground truth, and
runs any SV estimator (including a pluggable OddSHAP variant) on the paper's setup.

Design goals
------------
1. **Clean**: the notebook imports from here; no experiment logic lives in cells.
2. **Faithful to the paper**: interventional value function (50 background samples),
   exact interventional Shapley ground truth, the paper's eight value functions,
   budget = 100*d for Table 1, a log budget grid for Figure 2, fixed m=10,000 for
   the eta ablation.
3. **Pluggable**: ``estimator_factory`` accepts a name; ``"OddSHAP"`` resolves through
   the variant registry, so the *same* harness runs PR #522 vs PR #560 vs the installed
   library on identical games/seeds and the results are directly comparable.

Nothing here reads a precomputed CSV — every number is produced by these functions.
Heavy full-scale runs go through the cluster scripts; the notebook calls the same
functions at a small scale for a live, self-contained demo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import shapiq_games.datasets as datasets
from shapiq.approximator import (
    SVARM,
    KernelSHAP,
    PermutationSamplingSV,
    RegressionMSR,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq_games.benchmark.interventionaltreeshapiq_xai import InterventionalGame

from .oddshap_variants import load_oddshap, variant_label

# --- experiment constants (paper-aligned) ----------------------------------- #
RANDOM_STATE = 40
N_BACKGROUND = 50
ETAS = [50, 10, 5, 2]
ETA_BUDGET = 10_000
# the baseline sampling estimators we control (paper adds LeverageSHAP / 3-PolySHAP,
# which are other groups' methods and not importable here).
BASELINE_ESTIMATORS = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR"]


# csv-name, loader, kind, paper d — the six tabular value functions
TABULAR_VFS = [
    ("cancer", datasets.load_breast_cancer, "native_binary", 30),
    ("realestate", datasets.load_real_estate, "continuous", 15),
    ("corrgroups60", datasets.load_corrgroups60, "continuous", 60),
    ("independentlinear60", datasets.load_independentlinear60, "continuous", 60),
    ("nhanes", datasets.load_nhanesi, "continuous", 79),
    ("crime", datasets.load_communities_and_crime, "continuous", 101),
]


def make_estimator(name: str, n: int, *, oddshap_variant: str = "v522_merged"):
    """Build an SV estimator by name.

    ``name="OddSHAP"`` resolves through the variant registry (``oddshap_variant``),
    so the same call site runs any vendored revision. All others are stock shapiq.
    """
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
    if name == "OddSHAP":
        return load_oddshap(oddshap_variant)(n=n, random_state=0)
    msg = f"unknown estimator {name!r}"
    raise ValueError(msg)


def sfv(iv, n: int) -> np.ndarray:
    """Single-feature (order-1) Shapley values as a length-n vector."""
    return np.array([float(iv.dict_values.get((i,), 0.0)) for i in range(n)])


def safe_mse(est_name: str, n: int, budget: int, game, truth, *, oddshap_variant="v522_merged"):
    """Run one estimator; return Shapley MSE, or ``inf`` if it refuses the budget."""
    try:
        iv = make_estimator(est_name, n, oddshap_variant=oddshap_variant).approximate(budget, game)
        return float(np.mean((sfv(iv, n) - truth) ** 2))
    except (ValueError, RuntimeError):
        return float("inf")


def warm_dispatch() -> None:
    """Register the lightgbm tree converter in the main thread before any parallel use."""
    try:
        from lightgbm import LGBMRegressor

        from shapiq.tree.conversion import convert_tree_model

        r = np.random.default_rng(0)
        convert_tree_model(
            LGBMRegressor(n_estimators=2, max_depth=2, verbose=-1).fit(r.random((30, 3)), r.random(30))
        )
    except ImportError:
        pass


# --- value function + ground truth ------------------------------------------ #
@dataclass
class PreparedVF:
    name: str
    model: object
    background: np.ndarray
    explainer: object          # InterventionalTreeExplainer (exact interventional GT)
    n: int
    x_test: np.ndarray
    is_classifier: bool

    def game(self, target: np.ndarray) -> InterventionalGame:
        return InterventionalGame(
            model=self.model, reference_data=self.background, target_instance=target,
            class_index=1 if self.is_classifier else None,
        )

    def ground_truth(self, target: np.ndarray) -> np.ndarray:
        return sfv(self.explainer.explain_function(target.astype(np.float32)), self.n)


def prepare_vf(loader, kind: str, *, classifier: bool) -> PreparedVF:
    """Train the value-function model and build the exact interventional ground truth."""
    x, y = loader()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    is_clf = classifier
    if kind == "native_binary":
        y = y.astype(int)
        is_clf = True
    elif classifier:
        y = (y > np.median(y)).astype(int)
    n = x.shape[1]
    x_tr, x_te, y_tr, _ = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    model = (XGBClassifier if is_clf else XGBRegressor)(random_state=RANDOM_STATE, n_jobs=1)
    model.fit(x_tr, y_tr)
    rng = np.random.default_rng(RANDOM_STATE)
    bg = x_tr[rng.choice(x_tr.shape[0], size=min(N_BACKGROUND, x_tr.shape[0]), replace=False)]
    explainer = InterventionalTreeExplainer(
        model=model, data=bg.astype(np.float32), index="SV", max_order=1,
        class_index=1 if is_clf else None,
    )
    return PreparedVF(loader.__name__, model, bg, explainer, n, x_te, is_clf)


# --- experiments (return plain dicts; the notebook plots them) -------------- #
def table1_cell(vf: PreparedVF, n_instances: int, estimators, *, oddshap_variant="v522_merged"):
    """Per-estimator median / IQR of Shapley MSE at budget = 100*d."""
    budget = max(vf.n + 1, 100 * vf.n)
    n_use = min(n_instances, vf.x_test.shape[0])
    per = {est: [] for est in estimators}
    for i in range(n_use):
        target = vf.x_test[i]
        truth = vf.ground_truth(target)
        game = vf.game(target)
        for est in estimators:
            per[est].append(safe_mse(est, vf.n, budget, game, truth, oddshap_variant=oddshap_variant))
    return {
        est: {
            "median": float(np.median(per[est])),
            "q1": float(np.quantile(per[est], 0.25)),
            "q3": float(np.quantile(per[est], 0.75)),
            "std": float(np.std(per[est])),
            "n": n_use,
        }
        for est in estimators
    }


def fig2_curve(vf: PreparedVF, budgets, n_instances: int, estimators, *, oddshap_variant="v522_merged"):
    """Median Shapley MSE vs budget, per estimator."""
    n_use = min(n_instances, vf.x_test.shape[0])
    acc = {est: {b: [] for b in budgets} for est in estimators}
    for i in range(n_use):
        target = vf.x_test[i]
        truth = vf.ground_truth(target)
        game = vf.game(target)
        for b in budgets:
            for est in estimators:
                m = safe_mse(est, vf.n, b, game, truth, oddshap_variant=oddshap_variant)
                if np.isfinite(m):
                    acc[est][b].append(m)
    return {est: {b: float(np.median(v)) for b, v in acc[est].items() if v} for est in estimators}


def eta_ratios(vf: PreparedVF, n_instances: int, *, oddshap_variant="v522_merged"):
    """Figure-4 interaction-sparsity ablation: MSE ratio vs the interaction-free baseline."""
    OddSHAP = load_oddshap(oddshap_variant)
    n_use = min(n_instances, vf.x_test.shape[0])
    per = {e: [] for e in [*ETAS, "base"]}
    for i in range(n_use):
        target = vf.x_test[i]
        truth = vf.ground_truth(target)
        game = vf.game(target)
        for e in ETAS:
            try:
                iv = OddSHAP(n=vf.n, random_state=0, interaction_factor=e).approximate(ETA_BUDGET, game)
                per[e].append(float(np.mean((sfv(iv, vf.n) - truth) ** 2)))
            except (ValueError, RuntimeError):
                per[e].append(float("nan"))
        base = OddSHAP(n=vf.n, random_state=0, interaction_factor=10)
        base._select_odd_interactions = lambda **kw: []  # noqa: SLF001
        iv0 = base.approximate(ETA_BUDGET, game)
        per["base"].append(float(np.mean((sfv(iv0, vf.n) - truth) ** 2)))
    base_med = float(np.median(per["base"]))
    return {
        "n_interactions": [int(math.ceil(ETA_BUDGET / e)) for e in ETAS],
        "etas": ETAS,
        "ratios": [float(np.median(per[e])) / base_med if base_med else float("nan") for e in ETAS],
        "base_median": base_med,
    }


def log_budgets(n: int, n_points: int = 10, hi_cap: int = 20_000):
    """Log-spaced integer budgets from d+1 to min(2**d, hi_cap)."""
    hi = min(2 ** n, hi_cap)
    return sorted({int(round(b)) for b in np.logspace(np.log10(n + 1), np.log10(hi), n_points)})


__all__ = [
    "BASELINE_ESTIMATORS", "ETAS", "ETA_BUDGET", "TABULAR_VFS",
    "PreparedVF", "prepare_vf", "make_estimator", "sfv", "safe_mse", "warm_dispatch",
    "table1_cell", "fig2_curve", "eta_ratios", "log_budgets",
    "load_oddshap", "variant_label",
]
