"""Aggregate gpu_repro shard outputs into a table1-style CSV.

Reads the ``PARTIAL <vf> <estimator> <idx> <mse>`` lines emitted by the Slurm
array tasks and writes per-estimator median / IQR over the instances, ranked by
median MSE — the same columns as ``table1_n30_all_classifier.csv``.

Usage:  python aggregate_gpu.py <vf> "<glob-of-slurm-out>" <out_csv>
"""

from __future__ import annotations

import csv
import glob
import sys
from collections import defaultdict

import numpy as np

EST = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]


def main() -> None:
    vf, out_glob, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
    per_est: dict[str, list[float]] = defaultdict(list)
    for path in glob.glob(out_glob):
        with open(path) as fh:
            for line in fh:
                if line.startswith("PARTIAL"):
                    _, v, est, _idx, mse = line.split()
                    if v == vf:
                        per_est[est].append(float(mse))

    rows = []
    for est in EST:
        vals = np.asarray(per_est.get(est, []), dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            (
                vf,
                est,
                "exact_gt",
                int(vals.size),
                float(np.median(vals)),
                float(np.percentile(vals, 25)),
                float(np.percentile(vals, 75)),
            )
        )
    rows.sort(key=lambda r: r[4])  # rank by median MSE

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["value_function", "estimator", "config", "n_instances", "median_mse", "q1", "q3"]
        )
        writer.writerows(rows)
    ranks = " < ".join(f"{r[1]}({r[4]:.2e})" for r in rows)
    print(f"wrote {csv_path}: {len(rows)} estimators, {rows[0][3]} instances")
    print(f"  ranking: {ranks}")


if __name__ == "__main__":
    main()
