"""Compare an OddSHAP Table-1 reproduction run against the published paper.

Given a reproduction CSV produced by ``benchmark/oddshap_table1.py``, this
tool aggregates our measured MSE per estimator and prints it side-by-side
with the values reported in arXiv:2602.01399 Table 1 / Table 3, with a
per-estimator verdict.

Purpose
-------
The reproduction is a *correctness check* on our approximator
implementations. The four baseline estimators (MSR, SVARM,
PermutationSampling, RegressionMSR) are well-established shapiq
implementations: if our reproduction matches the paper on those, the
harness and protocol are sound — and then OddSHAP's number is
trustworthy. If OddSHAP matches too, OddSHAP is implemented correctly;
if OddSHAP is off while the baselines match, that isolates an OddSHAP
bug.

A reproduction on a consumer machine will not match the paper to three
significant figures — different hardware, RNG streams, instance set, and
exact token counts all move the number. The verdict is therefore
*order-of-magnitude*: a ratio within ~10x of the paper is "consistent"
(the implementation is sound); a ratio off by 100x+ is a "mismatch"
(a real bug, as seen when the OddSHAP Fourier-sign bug produced MSE ~0.5
instead of ~1e-5).

Run::

    python benchmark/compare_to_paper.py \\
        --csv benchmark/results/oddshap_table1.csv --game distilbert
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Published reference values — arXiv:2602.01399 Table 3 ("Expanded Table", p.12)
# Mean MSE per estimator per value function. Estimator names are normalised to
# this harness's labels (MSR / SVARM / PermutationSampling / RegressionMSR /
# OddSHAP plus the two not run here, LeverageSHAP / 3-PolySHAP).
# -----------------------------------------------------------------------------

PAPER_TABLE1_MEAN_MSE: dict[str, dict[str, float]] = {
    "distilbert": {  # DistilBERT, d = 14
        "MSR": 7.5e-4,
        "SVARM": 3.7e-4,
        "PermutationSampling": 6.2e-4,
        "3-PolySHAP": 8.0e-5,
        "LeverageSHAP": 7.7e-5,
        "RegressionMSR": 3.1e-5,
        "OddSHAP": 5.2e-5,
    },
    "vit16": {  # ViT16, d = 16
        "MSR": 1.2e-4,
        "SVARM": 5.7e-5,
        "PermutationSampling": 1.2e-4,
        "3-PolySHAP": 3.8e-5,
        "LeverageSHAP": 3.5e-5,
        "RegressionMSR": 1.0e-5,
        "OddSHAP": 1.5e-5,
    },
}

# Paper d per value function — used only to annotate the comparison.
PAPER_D: dict[str, int] = {"distilbert": 14, "vit16": 16}


# -----------------------------------------------------------------------------
# Read + aggregate our reproduction CSV
# -----------------------------------------------------------------------------


def load_reproduction(csv_path: Path) -> dict[str, list[float]]:
    """Return {method: [mse, ...]} from a benchmark/oddshap_table1.py CSV."""
    by_method: dict[str, list[float]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("mse", "")
            if value in ("", "nan", "NaN"):
                continue
            try:
                mse = float(value)
            except ValueError:
                continue
            if math.isnan(mse):
                continue
            by_method.setdefault(row["method"], []).append(mse)
    return by_method


def _verdict(ratio: float) -> str:
    """Order-of-magnitude verdict from our_mean / paper_mean."""
    if math.isnan(ratio):
        return "no data"
    if ratio <= 0:
        return "no data"
    log10 = abs(math.log10(ratio))
    if log10 <= 0.5:      # within ~3x
        return "MATCH"
    if log10 <= 1.0:      # within ~10x
        return "consistent"
    if log10 <= 2.0:      # within ~100x
        return "weak"
    return "MISMATCH"


# -----------------------------------------------------------------------------
# Comparison table
# -----------------------------------------------------------------------------


def compare(csv_path: Path, game: str) -> int:
    if game not in PAPER_TABLE1_MEAN_MSE:
        print(f"No paper reference for game '{game}'. "
              f"Known: {sorted(PAPER_TABLE1_MEAN_MSE)}")
        return 1

    reference = PAPER_TABLE1_MEAN_MSE[game]
    measured = load_reproduction(csv_path)
    n_instances = max((len(v) for v in measured.values()), default=0)

    print()
    print(f"OddSHAP Table 1 reproduction vs paper — {game} "
          f"(paper d = {PAPER_D.get(game, '?')})")
    print(f"Reproduction CSV: {csv_path}  ({n_instances} instances)")
    print()
    print(f"{'Estimator':<22} {'Paper mean':>12} {'Our mean':>12} "
          f"{'Our median':>12} {'Ratio':>9}  Verdict")
    print("-" * 84)

    for method, paper_mean in reference.items():
        our_values = measured.get(method, [])
        if not our_values:
            note = ("not run in this harness"
                    if method in ("3-PolySHAP", "LeverageSHAP")
                    else "no data")
            print(f"{method:<22} {paper_mean:>12.2e} {'-':>12} "
                  f"{'-':>12} {'-':>9}  {note}")
            continue
        our_mean = float(np.mean(our_values))
        our_median = float(np.median(our_values))
        ratio = our_mean / paper_mean
        print(f"{method:<22} {paper_mean:>12.2e} {our_mean:>12.2e} "
              f"{our_median:>12.2e} {ratio:>8.2f}x  {_verdict(ratio)}")

    print()
    print("Verdict scale (our_mean / paper_mean):")
    print("  MATCH       within ~3x   — implementation sound, protocol aligned")
    print("  consistent  within ~10x  — implementation sound, minor protocol drift")
    print("  weak        within ~100x — investigate (protocol or partial bug)")
    print("  MISMATCH    over 100x    — real bug")
    print()
    print("Baselines (MSR / SVARM / PermutationSampling / RegressionMSR) are")
    print("shapiq's own implementations: if they land MATCH/consistent the")
    print("harness is sound, so the OddSHAP row can be read as a verdict on")
    print("our OddSHAP implementation.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare an OddSHAP Table-1 reproduction CSV to the paper.",
    )
    parser.add_argument(
        "--csv", required=True, type=Path,
        help="Reproduction CSV from benchmark/oddshap_table1.py.",
    )
    parser.add_argument(
        "--game", default="distilbert", choices=sorted(PAPER_TABLE1_MEAN_MSE),
        help="Which paper value-function column to compare against.",
    )
    args = parser.parse_args(argv)
    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        return 1
    return compare(args.csv, args.game)


if __name__ == "__main__":
    raise SystemExit(main())
