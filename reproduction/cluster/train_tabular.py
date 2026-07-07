"""Comprehensive tabular training for the OddSHAP paper reproduction.

Produces, for the six tabular value functions and a chosen OddSHAP variant, the raw
per-instance data behind every tabular paper figure/table:

  * Table 1 / Table 3   -- MSE at budget = 100*d, per instance (median + full IQR + std)
  * Figure 2            -- MSE vs a log budget grid (d+1 .. min(2**d, 20000))
  * Figure 4 / Figure 11-- eta ablation at m in {5000, 10000, 20000}
  * Figure 5            -- wall-clock runtime vs budget (for the runtime figures)

Everything is written as tidy CSVs into the (gitignored) data directory, tagged by
value function and OddSHAP variant, so the notebooks can reproduce any paper panel.

Parallelism: exact interventional ground truth is precomputed serially (the tree
explainer is not safe to share across workers), then the per-instance estimator work
is dispatched to all cores via joblib/loky. Run one value function per array task to
spread across nodes.

Usage
-----
    python -m reproduction.cluster.train_tabular --vf cancer --variant v522_merged \
        --experiments table1 fig2 eta runtime --instances 30 --jobs -1 --out data
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from reproduction.core.constants import (
    ESTIMATORS, ETA_BUDGETS, ETAS, SCHEMA_ETA, SCHEMA_FIG2, SCHEMA_RUNTIME, SCHEMA_TABLE1,
    TABULAR_VF_NAMES, VARIANT_CHOICES,
)
from reproduction.core.harness import (
    TABULAR_VFS, interaction_free_oddshap, log_budgets, make_estimator, make_game,
    prepare_vf, sfv, warm_dispatch,
)


def _write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


# --- per-instance workers (picklable: receive arrays, not the tree explainer) --- #
def _mse_row(target, truth, model, bg, n, is_clf, budget, variant):
    from reproduction.core.harness import make_game  # local import for the loky worker
    game = make_game(model, bg, target, is_classifier=is_clf)
    out = {}
    for est in ESTIMATORS:
        try:
            iv = make_estimator(est, n, oddshap_variant=variant).approximate(budget, game)
            out[est] = float(np.mean((sfv(iv, n) - truth) ** 2))
        except (ValueError, RuntimeError):
            out[est] = float("inf")
    return out


def run_table1(vf, truths, n_instances, jobs, variant, out: Path):
    budget = max(vf.n + 1, 100 * vf.n)
    per = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_mse_row)(vf.x_test[i], truths[i], vf.model, vf.background, vf.n,
                          vf.is_classifier, budget, variant)
        for i in range(len(truths)))
    rows = []
    for est in ESTIMATORS:
        vals = np.array([p[est] for p in per], dtype=float)
        finite = vals[np.isfinite(vals)]
        rows.append((vf.name, est, variant, len(truths), budget,
                     float(np.median(finite)) if finite.size else float("inf"),
                     float(np.quantile(finite, 0.25)) if finite.size else float("inf"),
                     float(np.quantile(finite, 0.75)) if finite.size else float("inf"),
                     float(np.mean(finite)) if finite.size else float("inf"),
                     float(np.std(finite)) if finite.size else float("inf")))
    _write(out / f"table1_{vf.name}_{variant}.csv", SCHEMA_TABLE1, rows)


def run_fig2(vf, truths, jobs, variant, out: Path):
    budgets = log_budgets(vf.n)
    rows = []
    for b in budgets:
        per = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_mse_row)(vf.x_test[i], truths[i], vf.model, vf.background, vf.n,
                              vf.is_classifier, b, variant)
            for i in range(len(truths)))
        for est in ESTIMATORS:
            vals = np.array([p[est] for p in per], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                rows.append((vf.name, est, variant, b, len(truths),
                             float(np.median(finite)), float(np.quantile(finite, 0.25)),
                             float(np.quantile(finite, 0.75))))
    _write(out / f"fig2_{vf.name}_{variant}.csv", SCHEMA_FIG2, rows)


def _eta_row(target, truth, model, bg, n, is_clf, budget, variant):
    from reproduction.core.harness import interaction_free_oddshap, load_oddshap, make_game
    OddSHAP = load_oddshap(variant)
    game = make_game(model, bg, target, is_classifier=is_clf)
    out = {}
    for e in ETAS:
        try:
            iv = OddSHAP(n=n, random_state=0, interaction_factor=e).approximate(budget, game)
            out[e] = float(np.mean((sfv(iv, n) - truth) ** 2))
        except (ValueError, RuntimeError):
            out[e] = float("nan")
    try:
        iv0 = interaction_free_oddshap(n, oddshap_variant=variant).approximate(budget, game)
        out["base"] = float(np.mean((sfv(iv0, n) - truth) ** 2))
    except (ValueError, RuntimeError):
        out["base"] = float("nan")
    return out


def run_eta(vf, truths, jobs, variant, out: Path):
    rows = []
    for budget in ETA_BUDGETS:
        per = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_eta_row)(vf.x_test[i], truths[i], vf.model, vf.background, vf.n,
                              vf.is_classifier, budget, variant)
            for i in range(len(truths)))
        base_med = float(np.nanmedian([p["base"] for p in per]))
        for e in ETAS:
            med = float(np.nanmedian([p[e] for p in per]))
            rows.append((vf.name, variant, budget, e, int(np.ceil(budget / e)), len(truths),
                         med, med / base_med if base_med else float("nan")))
        rows.append((vf.name, variant, budget, "base", 0, len(truths), base_med, 1.0))
    _write(out / f"eta_{vf.name}_{variant}.csv", SCHEMA_ETA, rows)


def run_runtime(vf, truths, variant, out: Path):
    """Figure 5 — wall-clock runtime vs budget (median over a few instances, serial timing)."""
    budgets = log_budgets(vf.n)
    n_time = min(5, len(truths))
    rows = []
    for b in budgets:
        for est in ESTIMATORS:
            ts = []
            for i in range(n_time):
                game = vf.game(vf.x_test[i])
                t0 = time.perf_counter()
                try:
                    make_estimator(est, vf.n, oddshap_variant=variant).approximate(b, game)
                    ts.append(time.perf_counter() - t0)
                except (ValueError, RuntimeError):
                    pass
            if ts:
                rows.append((vf.name, est, variant, b, n_time, float(np.median(ts))))
    _write(out / f"runtime_{vf.name}_{variant}.csv", SCHEMA_RUNTIME, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", required=True, choices=TABULAR_VF_NAMES)
    ap.add_argument("--variant", default="v522_merged", choices=VARIANT_CHOICES)
    ap.add_argument("--experiments", nargs="+", default=["table1", "fig2", "eta", "runtime"],
                    choices=["table1", "fig2", "eta", "runtime"])
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="reproduction/data")
    args = ap.parse_args()

    warm_dispatch()
    loader, kind = next((l, k) for name, l, k, _ in TABULAR_VFS if name == args.vf)
    vf = prepare_vf(loader, kind, classifier=(kind != "native_binary"))
    vf.name = args.vf
    out = Path(args.out)
    n_use = min(args.instances, vf.x_test.shape[0])
    truths = [vf.ground_truth(vf.x_test[i]) for i in range(n_use)]  # serial, once
    print(f"{args.vf} d={vf.n} variant={args.variant} N={n_use} experiments={args.experiments}", flush=True)

    if "table1" in args.experiments:
        run_table1(vf, truths, n_use, args.jobs, args.variant, out)
    if "fig2" in args.experiments:
        run_fig2(vf, truths, args.jobs, args.variant, out)
    if "eta" in args.experiments and args.vf != "realestate":  # paper excludes Estate from Fig 4
        run_eta(vf, truths, args.jobs, args.variant, out)
    if "runtime" in args.experiments:
        run_runtime(vf, truths, args.variant, out)
    print(f"DONE {args.vf} {args.variant}", flush=True)


if __name__ == "__main__":
    main()
