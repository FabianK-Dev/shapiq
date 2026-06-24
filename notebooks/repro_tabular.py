"""Parallel tabular reproduction for OddSHAP (Table 1, Figure 2, Figure 4).

Regenerates the tabular `cluster_results/*.csv` against the canonical OddSHAP
(post-merge: candidate count ceil(m/eta) - d). The per-instance work is
embarrassingly parallel, so every (value-function, instance) is dispatched to a
joblib worker — set ``--jobs -1`` on a fat CPU node to use all cores.

Usage
-----
    python notebooks/repro_tabular.py --experiment all --jobs -1          # full scale
    python notebooks/repro_tabular.py --experiment table1 --instances 3   # quick local check

Outputs (into cluster_results/):
    table1_n30_all_classifier.csv          value_function,estimator,config,n_instances,median_mse,q1,q3
    table1_n30_estate_crime_regressor.csv  (same schema, Estate/Crime regressor reading)
    fig2_budget_curves_n10_classifier.csv  value_function,estimator,budget,n_instances,median_mse
    fig2_budget_curves_n10_regressor.csv   (Estate/Crime regressor reading)
    eta_ablation_n30_budget10000.csv       value_function,eta,n_interactions,n_instances,median_mse,mse_ratio_vs_interaction_free
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import shapiq_games.datasets as datasets
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
from shapiq_games.benchmark.interventionaltreeshapiq_xai import InterventionalGame

RANDOM_STATE = 40

def _warm_dispatch():
    """Register the lightgbm tree converter in the main process before parallel workers."""
    try:
        from lightgbm import LGBMRegressor
        from shapiq.tree.conversion import convert_tree_model
        r = np.random.default_rng(0)
        convert_tree_model(LGBMRegressor(n_estimators=2, max_depth=2, verbose=-1).fit(r.random((30, 3)), r.random(30)))
    except ImportError:
        pass

N_BACKGROUND = 50
ESTIMATORS = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]

OUT = Path("cluster_results") if Path("cluster_results").is_dir() else Path("notebooks/cluster_results")

# csv-name, loader, is_classifier (for the classifier reading)
VFS = [
    ("cancer", datasets.load_breast_cancer, "native_binary"),
    ("realestate", datasets.load_real_estate, "continuous"),
    ("corrgroups60", datasets.load_corrgroups60, "continuous"),
    ("independentlinear60", datasets.load_independentlinear60, "continuous"),
    ("nhanes", datasets.load_nhanesi, "continuous"),
    ("crime", datasets.load_communities_and_crime, "continuous"),
]


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


def sfv(iv, n: int) -> np.ndarray:
    return np.array([float(iv.dict_values.get((i,), 0.0)) for i in range(n)])


def _prepare(loader, kind: str, *, classifier: bool):
    """Train the value-function model and build the exact interventional GT."""
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
    gt = InterventionalTreeExplainer(
        model=model, data=bg.astype(np.float32), index="SV", max_order=1,
        class_index=1 if is_clf else None,
    )
    return model, bg, gt, n, x_te, is_clf


# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #
def _ground_truths(gt, x_te, n, n_use):
    """Exact interventional Shapley values, serial in the main process (the tree
    explainer is not safe to share across joblib workers)."""
    return [sfv(gt.explain_function(x_te[i].astype(np.float32)), n) for i in range(n_use)]


def _t1_instance(target, truth, model, bg, n, is_clf, budget):
    game = InterventionalGame(model=model, reference_data=bg, target_instance=target,
                              class_index=1 if is_clf else None)
    out = {}
    for est in ESTIMATORS:
        try:
            out[est] = float(np.mean((sfv(make_estimator(est, n).approximate(budget, game), n) - truth) ** 2))
        except Exception:  # noqa: BLE001
            out[est] = float("inf")
    return out


def run_table1(n_instances: int, jobs: int):
    rows_clf, rows_reg = [], []
    for vf, loader, kind in VFS:
        for classifier, sink, cfg in (
            (True, rows_clf, "xgb_classifier"),
            *([] if kind == "native_binary" else [(False, rows_reg, "xgb_regressor")]),
        ):
            model, bg, gt, n, x_te, is_clf = _prepare(loader, kind, classifier=classifier)
            budget = max(n + 1, 100 * n)
            n_use = min(n_instances, x_te.shape[0])
            truths = _ground_truths(gt, x_te, n, n_use)
            per = Parallel(n_jobs=jobs, backend="loky")(
                delayed(_t1_instance)(x_te[i], truths[i], model, bg, n, is_clf, budget) for i in range(n_use)
            )
            for est in ESTIMATORS:
                e = np.array([p[est] for p in per])
                sink.append((vf, est, cfg, n_use, float(np.median(e)),
                             float(np.quantile(e, 0.25)), float(np.quantile(e, 0.75))))
            print(f"table1 {vf:20s} {cfg:14s} d={n:3d} N={n_use} done", flush=True)
    _write(OUT / "table1_n30_all_classifier.csv",
           ["value_function", "estimator", "config", "n_instances", "median_mse", "q1", "q3"], rows_clf)
    _write(OUT / "table1_n30_estate_crime_regressor.csv",
           ["value_function", "estimator", "config", "n_instances", "median_mse", "q1", "q3"],
           [r for r in rows_reg if r[0] in ("realestate", "crime")])


# --------------------------------------------------------------------------- #
# Figure 2 — budget curves
# --------------------------------------------------------------------------- #
def _f2_instance(target, truth, model, bg, n, is_clf, budgets):
    game = InterventionalGame(model=model, reference_data=bg, target_instance=target,
                              class_index=1 if is_clf else None)
    out: dict[int, dict[str, float]] = {}
    for b in budgets:
        out[b] = {}
        for est in ESTIMATORS:
            try:
                iv = make_estimator(est, n).approximate(b, game)
            except (ValueError, RuntimeError):
                continue
            out[b][est] = float(np.mean((sfv(iv, n) - truth) ** 2))
    return out


def run_fig2(n_instances: int, jobs: int):
    rows_clf, rows_reg = [], []
    for vf, loader, kind in VFS:
        for classifier, sink in ((True, rows_clf),
                                 *([] if kind == "native_binary" else [(False, rows_reg)])):
            if not classifier and vf not in ("realestate", "crime"):
                continue
            model, bg, gt, n, x_te, is_clf = _prepare(loader, kind, classifier=classifier)
            hi = min(2 ** n, 20_000)
            budgets = sorted({int(round(b)) for b in np.logspace(np.log10(n + 1), np.log10(hi), 10)})
            n_use = min(n_instances, x_te.shape[0])
            truths = _ground_truths(gt, x_te, n, n_use)
            per = Parallel(n_jobs=jobs, backend="loky")(
                delayed(_f2_instance)(x_te[i], truths[i], model, bg, n, is_clf, budgets) for i in range(n_use)
            )
            for est in ESTIMATORS:
                for b in budgets:
                    vals = [p[b][est] for p in per if est in p[b]]
                    if vals:
                        sink.append((vf, est, b, len(vals), float(np.median(vals))))
            print(f"fig2 {vf:20s} {'reg' if not classifier else 'clf'} d={n:3d} N={n_use} done", flush=True)
    _write(OUT / "fig2_budget_curves_n10_classifier.csv",
           ["value_function", "estimator", "budget", "n_instances", "median_mse"], rows_clf)
    _write(OUT / "fig2_budget_curves_n10_regressor.csv",
           ["value_function", "estimator", "budget", "n_instances", "median_mse"], rows_reg)


# --------------------------------------------------------------------------- #
# Figure 4 — eta ablation
# --------------------------------------------------------------------------- #
ETAS = [50, 10, 5, 2]
ETA_BUDGET = 10_000


def _eta_instance(target, truth, model, bg, n, is_clf):
    game = InterventionalGame(model=model, reference_data=bg, target_instance=target,
                              class_index=1 if is_clf else None)
    out = {}
    for e in ETAS:
        iv = OddSHAP(n=n, random_state=0, interaction_factor=e).approximate(ETA_BUDGET, game)
        out[e] = float(np.mean((sfv(iv, n) - truth) ** 2))
    # interaction-free baseline: empty higher-order support (matches the gallery
    # script's plot_oddshap_eta_ablation.py exactly)
    base_est = OddSHAP(n=n, random_state=0, interaction_factor=10)
    base_est._select_odd_interactions = lambda **kw: []  # noqa: SLF001
    iv0 = base_est.approximate(ETA_BUDGET, game)
    out["base"] = float(np.mean((sfv(iv0, n) - truth) ** 2))
    return out


def run_eta(n_instances: int, jobs: int):
    rows = []
    for vf, loader, kind in VFS:
        # paper Section 5.3 / Figure 4 excludes Estate ("omitted due to outlier
        # improvements"); the 2 deep-learning value functions are run on GPU separately
        if vf == "realestate":
            continue
        # paper uses the classifier reading for the eta ablation
        classifier = kind != "native_binary"
        model, bg, gt, n, x_te, is_clf = _prepare(loader, kind, classifier=classifier)
        n_use = min(n_instances, x_te.shape[0])
        truths = _ground_truths(gt, x_te, n, n_use)
        per = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_eta_instance)(x_te[i], truths[i], model, bg, n, is_clf) for i in range(n_use)
        )
        base = float(np.median([p["base"] for p in per]))
        rows.append((vf, "base", "", n_use, base, 1.0))
        for e in ETAS:
            med = float(np.median([p[e] for p in per]))
            rows.append((vf, e, int(math.ceil(ETA_BUDGET / e)), n_use, med, med / base if base else 0.0))
        print(f"eta {vf:20s} d={n:3d} N={n_use} done", flush=True)
    _write(OUT / "eta_ablation_n30_budget10000.csv",
           ["value_function", "eta", "n_interactions", "n_instances", "median_mse",
            "mse_ratio_vs_interaction_free"], rows)


def _write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["table1", "fig2", "eta", "all"], default="all")
    ap.add_argument("--instances", type=int, default=30, help="N per VF (table1/eta); fig2 caps at this too")
    ap.add_argument("--jobs", type=int, default=-1, help="joblib n_jobs (-1 = all cores)")
    ap.add_argument("--vf", default=None, help="restrict to one value function (array sharding)")
    a = ap.parse_args()
    if a.vf:
        global VFS
        VFS = [v for v in VFS if v[0] == a.vf]
    _warm_dispatch()
    if a.experiment in ("table1", "all"):
        run_table1(a.instances, a.jobs)
    if a.experiment in ("fig2", "all"):
        run_fig2(min(a.instances, 10), a.jobs)
    if a.experiment in ("eta", "all"):
        run_eta(a.instances, a.jobs)


if __name__ == "__main__":
    main()
