"""Assemble the OddSHAP paper Table 1 from per-value-function reproduction CSVs.

Reads the ``oddshap_table1_<game>.csv`` files produced by
``benchmark/oddshap_table1.py`` (single-budget mode) and assembles the
Table-1 grid of arXiv:2602.01399: value functions as columns, estimators as
rows, each cell the mean MSE over the local-explanation instances at the
largest evaluated budget (m ~= 100d).

The bottom rows give, per column, the rank of each estimator (1 = lowest
MSE) and the average rank — the paper's headline metric. The paper's
conclusion is that OddSHAP attains the best (lowest) average rank; this
script reports whether the reproduction does too.

Run::

    python benchmark/build_table1.py --results-dir documents/oddshap_paper
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# Column order = paper Table 1 order (by dimension d). Image column ViT16 is
# omitted: the shapiq ViT setup is incompatible with transformers 5.x.
COLUMN_ORDER = [
    ("distilbert", "DistilBERT", 14),
    ("realestate", "Estate", 15),
    ("cancer", "Cancer", 30),
    ("corrgroups60", "CG60", 60),
    ("independentlinear60", "IL60", 60),
    ("nhanes", "NHANES", 79),
    ("crime", "Crime", 101),
]

ROW_ORDER = ["MSR", "SVARM", "PermutationSampling", "OddSHAP"]


def load_game_csv(path: Path) -> dict[str, float]:
    """Mean MSE per method at the largest budget in one reproduction CSV."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    budgets = [int(r["budget"]) for r in rows]
    top = max(budgets)
    by_method: dict[str, list[float]] = {}
    for r in rows:
        if int(r["budget"]) != top:
            continue
        value = r["mse"]
        if value in ("", "nan", "NaN"):
            continue
        try:
            mse = float(value)
        except ValueError:
            continue
        if not math.isnan(mse):
            by_method.setdefault(r["method"], []).append(mse)
    return {m: float(np.mean(v)) for m, v in by_method.items() if v}


def build(results_dir: Path) -> int:
    # game key -> {method: mean MSE}
    grid: dict[str, dict[str, float]] = {}
    for key, _, _ in COLUMN_ORDER:
        for candidate in (f"oddshap_table1_{key}.csv",
                          f"oddshap_table1_{key}_gpu.csv",
                          f"oddshap_table1_{key}_gpu30.csv"):
            path = results_dir / candidate
            if path.exists():
                grid[key] = load_game_csv(path)
                break

    present = [(k, label, d) for k, label, d in COLUMN_ORDER if grid.get(k)]
    if not present:
        print(f"No reproduction CSVs found in {results_dir}")
        return 1

    # ---- MSE grid -----------------------------------------------------------
    print()
    print("OddSHAP paper Table 1 reproduction — mean MSE at m ~= 100d")
    header = f"{'Estimator':<20}" + "".join(
        f"{label:>13}" for _, label, _ in present)
    print(header)
    print(f"{'(d)':<20}" + "".join(f"{d:>13}" for _, _, d in present))
    print("-" * len(header))
    for method in ROW_ORDER:
        cells = []
        for key, _, _ in present:
            mse = grid[key].get(method)
            cells.append(f"{mse:>13.2e}" if mse is not None else f"{'-':>13}")
        print(f"{method:<20}" + "".join(cells))

    # ---- ranks --------------------------------------------------------------
    print("-" * len(header))
    ranks: dict[str, list[int]] = {m: [] for m in ROW_ORDER}
    for key, _, _ in present:
        col = {m: grid[key].get(m) for m in ROW_ORDER}
        ordered = sorted(
            (m for m in ROW_ORDER if col[m] is not None),
            key=lambda m: col[m],
        )
        for rank, m in enumerate(ordered, start=1):
            ranks[m].append(rank)
    for method in ROW_ORDER:
        cells = []
        for i, (key, _, _) in enumerate(present):
            col = {m: grid[key].get(m) for m in ROW_ORDER}
            ordered = sorted(
                (m for m in ROW_ORDER if col[m] is not None),
                key=lambda m: col[m],
            )
            rank = ordered.index(method) + 1 if method in ordered else None
            cells.append(f"{rank:>13}" if rank else f"{'-':>13}")
        print(f"{'rank ' + method:<20}" + "".join(cells))

    print("-" * len(header))
    avg_ranks = {m: (float(np.mean(ranks[m])) if ranks[m] else float("nan"))
                 for m in ROW_ORDER}
    for method in ROW_ORDER:
        print(f"{'avg rank ' + method:<20}{avg_ranks[method]:>13.2f}")

    # ---- verdict ------------------------------------------------------------
    print()
    best = min(avg_ranks, key=lambda m: avg_ranks[m])
    odd_rank = avg_ranks["OddSHAP"]
    n_cols = len(present)
    odd_wins = sum(1 for key, _, _ in present
                   if grid[key].get("OddSHAP") is not None
                   and grid[key]["OddSHAP"] == min(
                       v for v in (grid[key].get(m) for m in ROW_ORDER)
                       if v is not None))
    print(f"Columns reproduced: {n_cols}/7 "
          f"(ViT16 omitted — transformers incompatibility)")
    print(f"OddSHAP average rank: {odd_rank:.2f}  "
          f"(best possible = 1.00, baselines = {len(ROW_ORDER)})")
    print(f"OddSHAP is rank-1 (lowest MSE) on {odd_wins}/{n_cols} columns")
    if best == "OddSHAP":
        print("VERDICT: OddSHAP attains the best average rank — reproduces "
              "the paper's headline Table 1 conclusion.")
    else:
        print(f"VERDICT: best average rank is '{best}', not OddSHAP — "
              "investigate.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the OddSHAP Table 1 from reproduction CSVs.",
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Directory holding oddshap_table1_<game>.csv files.",
    )
    args = parser.parse_args(argv)
    return build(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
