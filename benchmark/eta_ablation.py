"""OddSHAP interaction-count ablation — paper Figures 4 / 11.

OddSHAP's ``interaction_factor`` sets how many odd higher-order interactions
the proxy screening adds to the constrained Fourier regression basis:
``n_candidate_interactions = ceil(budget / interaction_factor)``. A *large*
factor therefore yields *few* interactions, a small factor yields many; the
paper's default is interaction_factor = 10.

The paper's Figures 4 and 11 study how MSE changes as the number of odd
interactions grows. This script sweeps ``interaction_factor`` from large
(few interactions) to small (many), runs OddSHAP at each, computes MSE
against exact ground truth, and reports the mean MSE plus the ratio against
the fewest-interactions setting — the paper's Figure-4 quantity. The paper's
conclusion is that adding odd interactions lowers MSE, sharply at first then
with diminishing returns.

Run::

    python benchmark/eta_ablation.py --game cancer --instances 10
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from oddshap_table1 import (
    GAME_FACTORIES,
    ODDSHAP_PROXY_MAX_ORDER,
    budget_grid,
    exact_ground_truth,
    _singletons,
    mse,
)
from shapiq.approximator.regression.oddshap import OddSHAP

# interaction_factor sweep, large -> small = few -> many odd interactions.
# 10 is the paper's default. The largest entry is the fewest-interactions
# reference; the smallest yields the most odd interactions. (Very large
# factors are dropped: OddSHAP raises "budget too small" when too few
# candidate interactions leave the regression system under-budgeted.)
FACTOR_GRID = [100, 40, 20, 10, 5, 2, 1]


def run(game_name: str, n_instances: int, seed: int, out_csv: Path) -> int:
    factory = GAME_FACTORIES[game_name]
    # interaction_factor -> list of per-instance MSE
    by_factor: dict[int, list[float]] = {f: [] for f in FACTOR_GRID}
    # interaction_factor -> list of realised odd-interaction counts
    by_count: dict[int, list[int]] = {f: [] for f in FACTOR_GRID}
    rows: list[tuple] = []

    for i, inst in enumerate(factory(n_instances, None), start=1):
        n = inst.n
        budget = budget_grid(n, grid=False)[0]
        exact = exact_ground_truth(inst.game, n)
        for factor in FACTOR_GRID:
            count = float("nan")
            try:
                estimator = OddSHAP(
                    n=n, random_state=seed, interaction_factor=factor,
                    proxy_max_order=min(ODDSHAP_PROXY_MAX_ORDER, n),
                )
                iv = estimator.approximate(budget, inst.game)
                value = mse(_singletons(iv, n), exact)
                count = int(getattr(estimator, "n_active_interactions", 0))
            except (ValueError, RuntimeError, MemoryError, KeyError,
                    IndexError, TypeError, ZeroDivisionError):
                value = float("nan")
            if not math.isnan(value):
                by_factor[factor].append(value)
            if not math.isnan(count):
                by_count[factor].append(count)
            rows.append((inst.label, n, factor, count, value))
        print(f"  [{i}/{n_instances}] {inst.label} n={n} "
              f"factor-grid done", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "n", "interaction_factor",
                         "n_active_interactions", "mse"])
        writer.writerows(rows)

    # ---- summary -----------------------------------------------------------
    print()
    print(f"OddSHAP interaction-count ablation — {game_name}")
    print(f"{'factor':>8} {'~interactions':>14} {'mean MSE':>14} "
          f"{'MSE ratio':>11} {'n_ok':>6}")
    print("-" * 58)
    # fewest-interactions reference = largest factor that produced data
    base_factor = next((f for f in FACTOR_GRID if by_factor[f]), FACTOR_GRID[0])
    base = (float(np.mean(by_factor[base_factor]))
            if by_factor[base_factor] else float("nan"))
    for factor in FACTOR_GRID:
        vals = by_factor[factor]
        mean = float(np.mean(vals)) if vals else float("nan")
        ratio = (mean / base) if base and not math.isnan(base) else float("nan")
        counts = by_count[factor]
        avg_count = (float(np.mean(counts)) if counts else float("nan"))
        print(f"{factor:>8} {avg_count:>14.1f} {mean:>14.3e} "
              f"{ratio:>11.3f} {len(vals):>6}")
    print()
    if not math.isnan(base):
        valid = [f for f in FACTOR_GRID if by_factor[f]]
        best_factor = min(valid, key=lambda f: np.mean(by_factor[f]))
        improved = base / float(np.mean(by_factor[best_factor]))
        print(f"Best interaction_factor = {best_factor}: {improved:.1f}x lower "
              f"MSE than the fewest-interactions setting (factor={base_factor}).")
        if improved > 1.05:
            print("VERDICT: adding odd interactions lowers MSE — reproduces "
                  "the paper's Figure-4 ablation conclusion.")
    print(f"\nCSV written: {out_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OddSHAP eta (interaction_factor) ablation.",
    )
    parser.add_argument("--game", default="cancer", choices=sorted(GAME_FACTORIES))
    parser.add_argument("--instances", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--output", default="benchmark/results/oddshap_eta_ablation.csv",
        type=Path,
    )
    args = parser.parse_args(argv)
    return run(args.game, args.instances, args.seed, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
