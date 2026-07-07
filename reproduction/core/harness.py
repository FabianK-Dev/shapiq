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

This module owns the *building blocks* — value function, ground truth, estimator
construction, the interaction-free baseline, and the budget grid. The cluster scripts
(`reproduction/cluster/`) compose them into the parallel experiment sweeps that write the
CSVs; the notebooks read those CSVs and plot. Small live demos (e.g. the variant delta,
the semivalue check) call these building blocks directly.
"""

from __future__ import annotations

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

from .constants import (  # noqa: F401  (re-exported for callers)
    BASELINE_ESTIMATORS, DEFAULT_VARIANT, ESTIMATORS, ETA_BUDGETS, ETAS,
    INTERACTION_FREE_FACTOR,
)
from .variants import load_oddshap, variant_label

# --- experiment constants (paper-aligned) ----------------------------------- #
RANDOM_STATE = 40
N_BACKGROUND = 50
ETA_BUDGET = 10_000  # Figure 4 headline budget (Figure 11 sweeps ETA_BUDGETS)


# csv-name, loader, kind, paper d — the six tabular value functions
TABULAR_VFS = [
    ("cancer", datasets.load_breast_cancer, "native_binary", 30),
    ("realestate", datasets.load_real_estate, "continuous", 15),
    ("corrgroups60", datasets.load_corrgroups60, "continuous", 60),
    ("independentlinear60", datasets.load_independentlinear60, "continuous", 60),
    ("nhanes", datasets.load_nhanesi, "continuous", 79),
    ("crime", datasets.load_communities_and_crime, "continuous", 101),
]


def make_estimator(name: str, n: int, *, oddshap_variant: str = DEFAULT_VARIANT):
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


def safe_mse(est_name: str, n: int, budget: int, game, truth, *, oddshap_variant=DEFAULT_VARIANT):
    """Run one estimator; return Shapley MSE, or ``inf`` if it refuses the budget.

    A budget an estimator refuses by contract (OddSHAP below its minimum) surfaces as a
    ``ValueError``/``RuntimeError`` and is recorded as ``inf`` (ranks last, filtered by
    ``np.isfinite`` downstream) rather than aborting a whole sweep.
    """
    try:
        iv = make_estimator(est_name, n, oddshap_variant=oddshap_variant).approximate(budget, game)
        return float(np.mean((sfv(iv, n) - truth) ** 2))
    except (ValueError, RuntimeError):
        return float("inf")


def interaction_free_oddshap(n: int, *, oddshap_variant: str = DEFAULT_VARIANT):
    """OddSHAP with an emptied higher-order support — the Figure-4 interaction-free baseline.

    The interaction_factor is irrelevant (the support is emptied), so it uses the shared
    ``INTERACTION_FREE_FACTOR`` constant. Centralised here so the monkey-patch lives in one
    place instead of being copied across the cluster scripts.
    """
    est = load_oddshap(oddshap_variant)(n=n, random_state=0, interaction_factor=INTERACTION_FREE_FACTOR)
    est._select_odd_interactions = lambda **kw: []  # noqa: SLF001
    return est


def make_game(model, background, target, *, is_classifier: bool) -> "InterventionalGame":
    """Build the interventional value function for one target instance.

    A free function (not a ``PreparedVF`` method) so the parallel workers can construct a
    game from only the picklable model/background/target, without carrying the tree
    explainer (which is not fork-safe).
    """
    return InterventionalGame(
        model=model, reference_data=background, target_instance=target,
        class_index=1 if is_classifier else None,
    )


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
        return make_game(self.model, self.background, target, is_classifier=self.is_classifier)

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


def log_budgets(n: int, n_points: int = 10, hi_cap: int = 20_000):
    """Log-spaced integer budgets from d+1 to min(2**d, hi_cap) — the paper's Fig-2 grid."""
    hi = min(2 ** n, hi_cap)
    return sorted({int(round(b)) for b in np.logspace(np.log10(n + 1), np.log10(hi), n_points)})


__all__ = [
    "BASELINE_ESTIMATORS", "ESTIMATORS", "ETAS", "ETA_BUDGET", "ETA_BUDGETS",
    "DEFAULT_VARIANT", "INTERACTION_FREE_FACTOR", "TABULAR_VFS",
    "PreparedVF", "prepare_vf", "make_estimator", "make_game", "interaction_free_oddshap",
    "sfv", "safe_mse", "warm_dispatch", "log_budgets",
    "load_oddshap", "variant_label",
]
