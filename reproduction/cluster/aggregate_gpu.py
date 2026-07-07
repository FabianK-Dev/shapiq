"""Aggregate GPU shard logs (train_gpu.py output) into the same tidy CSVs the tabular
trainer writes, so the notebooks read one uniform schema for all eight value functions.

Parses PARTIAL_T1 / PARTIAL_F2 / PARTIAL_ETA lines and writes, per (vf, variant):
    table1_<vf>_<variant>.csv   fig2_<vf>_<variant>.csv   eta_<vf>_<variant>.csv

Usage:  python -m reproduction.cluster.aggregate_gpu --vf vit16 --variant v522_merged \
            --logs "reproduction/data/gpu_vit16_v522_merged.log" --out reproduction/data
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

EST = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]
ETAS = [50, 10, 5, 2]


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--logs", required=True, help="glob of shard logs")
    ap.add_argument("--out", default="reproduction/data")
    args = ap.parse_args()
    out = Path(args.out)

    t1 = defaultdict(list)                       # est -> [mse]
    f2 = defaultdict(lambda: defaultdict(list))  # est -> budget -> [mse]
    et = defaultdict(lambda: defaultdict(list))  # key(eta|base) -> budget -> [mse]
    t1_budget = None

    for path in glob.glob(args.logs):
        with open(path) as fh:
            for line in fh:
                p = line.split()
                if len(p) < 5 or p[1] != args.vf:
                    continue
                if p[0] == "PARTIAL_T1":
                    _, _, est, _i, budget, mse = p[:6]
                    t1[est].append(float(mse)); t1_budget = int(budget)
                elif p[0] == "PARTIAL_F2":
                    _, _, est, _i, budget, mse = p[:6]
                    f2[est][int(budget)].append(float(mse))
                elif p[0] == "PARTIAL_ETA":
                    _, _, key, _i, budget, mse = p[:6]
                    et[key][int(budget)].append(float(mse))

    # Table 1
    rows = []
    for est in EST:
        v = np.array(t1.get(est, []), dtype=float)
        if v.size:
            rows.append((args.vf, est, args.variant, v.size, t1_budget,
                         float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75)),
                         float(np.mean(v)), float(np.std(v))))
    if rows:
        _write(out / f"table1_{args.vf}_{args.variant}.csv",
               ["vf", "estimator", "variant", "n", "budget", "median", "q1", "q3", "mean", "std"], rows)

    # Figure 2
    rows = []
    for est in EST:
        for b, vals in sorted(f2.get(est, {}).items()):
            v = np.array(vals, dtype=float)
            rows.append((args.vf, est, args.variant, b, v.size,
                         float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75))))
    if rows:
        _write(out / f"fig2_{args.vf}_{args.variant}.csv",
               ["vf", "estimator", "variant", "budget", "n", "median", "q1", "q3"], rows)

    # Figure 4/11 eta
    rows = []
    budgets = sorted({b for d in et.values() for b in d})
    for budget in budgets:
        base_med = float(np.nanmedian(et.get("base", {}).get(budget, [np.nan])))
        for e in ETAS:
            vals = et.get(str(e), {}).get(budget, [])
            if vals:
                med = float(np.nanmedian(vals))
                rows.append((args.vf, args.variant, budget, e, int(np.ceil(budget / e)), len(vals),
                             med, med / base_med if base_med else float("nan")))
        if et.get("base", {}).get(budget):
            rows.append((args.vf, args.variant, budget, "base", 0, len(et["base"][budget]), base_med, 1.0))
    if rows:
        _write(out / f"eta_{args.vf}_{args.variant}.csv",
               ["vf", "variant", "budget", "eta", "n_interactions", "n", "median_mse", "ratio_vs_base"], rows)


if __name__ == "__main__":
    main()
