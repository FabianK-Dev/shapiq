"""Aggregate gpu_repro shard outputs into Table-1 and Figure-4 (eta) CSVs.

Reads the lines emitted by the Slurm array tasks:
    PARTIAL     <vf> <estimator>   <idx> <mse>     -> Table 1 (median / IQR per estimator)
    PARTIAL_ETA <vf> <eta|base>    <idx> <mse>     -> Figure 4 (MSE ratio vs interaction-free)
(PARTIAL_F2 lines, if present, are ignored — Figure 2 uses the paper's extracted GPU curves.)

Usage:  python aggregate_gpu.py <vf> "<glob-of-slurm-out>" <table1_csv> <eta_csv>
"""

from __future__ import annotations

import csv
import glob
import math
import sys
from collections import defaultdict

import numpy as np

EST = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]
ETAS = [50, 10, 5, 2]
ETA_BUDGET = 10_000


def main() -> None:
    vf, out_glob, t1_path, eta_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    per_est: dict[str, list[float]] = defaultdict(list)
    per_eta: dict[str, list[float]] = defaultdict(list)  # key: "50","10","5","2","base"
    for path in glob.glob(out_glob):
        with open(path) as fh:
            for line in fh:
                tok = line.split()
                if line.startswith("PARTIAL_ETA "):
                    _, v, key, _idx, mse = tok
                    if v == vf:
                        per_eta[key].append(float(mse))
                elif line.startswith("PARTIAL ") and len(tok) == 5:
                    _, v, est, _idx, mse = tok
                    if v == vf:
                        per_est[est].append(float(mse))

    # --- Table 1 ---
    rows = []
    for est in EST:
        vals = np.asarray(per_est.get(est, []), dtype=float)
        if vals.size == 0:
            continue
        rows.append((vf, est, "exact_gt", int(vals.size),
                     float(np.median(vals)), float(np.percentile(vals, 25)), float(np.percentile(vals, 75))))
    rows.sort(key=lambda r: r[4])
    with open(t1_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["value_function", "estimator", "config", "n_instances", "median_mse", "q1", "q3"])
        w.writerows(rows)
    if rows:
        print(f"wrote {t1_path}: {len(rows)} estimators, {rows[0][3]} instances")
        print("  ranking: " + " < ".join(f"{r[1]}({r[4]:.2e})" for r in rows))

    # --- Figure 4 eta ---
    eta_rows = []
    if per_eta.get("base"):
        base = float(np.median(per_eta["base"]))
        n_use = len(per_eta["base"])
        eta_rows.append((vf, "base", "", n_use, base, 1.0))
        for e in ETAS:
            vals = per_eta.get(str(e), [])
            if not vals:
                continue
            med = float(np.median(vals))
            eta_rows.append((vf, e, int(math.ceil(ETA_BUDGET / e)), n_use, med, med / base if base else 0.0))
        with open(eta_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["value_function", "eta", "n_interactions", "n_instances", "median_mse",
                        "mse_ratio_vs_interaction_free"])
            w.writerows(eta_rows)
        ratios = [f"eta={r[1]}:{r[5]:.3f}" for r in eta_rows if r[1] != "base"]
        print(f"wrote {eta_path}: {ratios}")


if __name__ == "__main__":
    main()
