# Cross-method sweep — 2026-07-17 (LMU CIP cluster)

Every SV approximator registered in `shapiq.approximator.SV_APPROXIMATORS`, run on a shared set
of games over a budget grid, ten predictions per game. This is the run the cross-method
discussion (task 4) is written against.

## How it was generated

The sweep is `benchmark/performance.py`, nothing else. It is equivalent to this single command:

```bash
python -m benchmark.performance \
    --budget-mults 1,2,3,5,8,12,20,30,50,75,110,160 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --plot
```

It was run as ten cluster jobs instead, one per repeat, purely to cut wall-clock time. Repeats
are independent — the model is fixed, and a repeat only selects which held-out prediction to
explain — so running them in ten processes and concatenating the CSVs yields the same cells as
one process running all ten. Per job:

```bash
python -m benchmark.performance \
    --budget-mults 1,2,3,5,8,12,20,30,50,75,110,160 \
    --seeds "$SEED" \
    --name "full_seed${SEED}" \
    --output-root ~/bench_full/results
```

`--plot` was not passed to the jobs: ten single-repeat plot sets would each show a band over one
prediction. The figures here were produced afterwards from the merged `results.csv` via
`plot_all_figures(results, out_dir, style_name="default")`, which is the same function `--plot`
calls.

| | |
|---|---|
| Code | `3f5c4765` (branch `submission`) |
| Where | LMU CIP cluster, `krater20`, Slurm job `200711`, array 0-9 |
| Nodes | 14 cores per task, ten nodes in parallel |
| Wall-clock | 23-27 min per task |
| Date | 2026-07-17 |

## What a repeat is

A repeat explains **one held-out prediction of one fixed model**. The split and the model are
pinned (`MODEL_SEED`), and the repeat index selects the prediction; for the SOUM games, which
have no predictions, it draws the random game. It also seeds the estimator.

This matters for reading the bands. If a repeat rebuilt the model, the spread across repeats
would mix estimator noise with model noise and could be read as neither — a different tree has
different Shapley values, so the target would move with it. Fixing the model makes the band the
spread across predictions, which is what the approximator papers report.

## Why `--budget-mults` and not the default `--budgets`

`--budgets` are fractions of the 2^n coalition space. That only expresses a runnable budget while
n is small: at n=60 even 5% is 5.8e16 evaluations. It also saturates — at 100% of 2^n every
method is exact, so the ranks there are ties, not a comparison.

`--budget-mults` sizes the budget as `mult * n`, which is the convention the approximator papers
use, applies to every game including the large ones, and puts the large games in the
under-determined regime where the methods actually differ. On `Communities(n=101)` the grid spans
101 to 16,160 evaluations, i.e. `1*d` to `160*d`, which reaches the order of the OddSHAP paper's
cap of `min(2^d, 20000)`.

## What ran

* **Games (11).** `IRIS(n=4)`, `SOUM(n=6)`, `California(n=8)`, `SOUM(n=8)`, `Diabetes(n=10)`,
  `SOUM(n=10)`, `Adult(n=12)`, `Correlated(n=60)`, `Independent(n=60)`, `NHANES(n=79)`,
  `Communities(n=101)`. The ML games follow the eight-dataset grid of Musco & Witter (2024).
* **Methods (15 registered, 14 ran).** `ShaplEIG` is skipped everywhere: it needs the `shapleig`
  extra (torch, gpytorch, botorch), which is not installed. That is an environment gap, not a
  defect — the rows record it as `skipped:incompatible_constructor(ImportError)`.
* **Budgets (12).** `mult * n` for `mult` in 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 110, 160.
* **Repeats (10).** Predictions 0-9 of the held-out split, model fixed.
* **Ground truth.** Games up to ten players enumerate the game exactly (`ExactComputer`). Larger
  games use interventional TreeSHAP against the same single baseline row the game masks with, so
  the target is the Shapley value of the game the approximators are actually given.

## Contents

| File | What |
|---|---|
| `results.csv` | 143,980 metric rows over 19,800 cells (method x game x budget x repeat), long format |
| `seed_*.csv` | the ten per-repeat CSVs the jobs wrote, before merging |
| `plots/` | 99 figures: nine metrics across the eleven games, all fifteen methods |
| `plots_paper_subset/` | the same, restricted to the eight methods the OddSHAP paper compares |

Status counts: 141,920 `ok`, 1,320 `skipped:incompatible_constructor` (all `ShaplEIG`), 740
`skipped:refused_regime` (methods that decline a budget below their minimum — OddSHAP, SPEX and
ProxySPEX do this by design rather than silently returning a degraded estimate).

## Reading the figures

* Each point is the **median over the ten predictions**, and the band is the **interquartile
  range** across them.
* Line style says which family a curve belongs to: **solid** for the three approximators this
  project contributed, **dashed** for the baselines. Colour and marker identify the individual
  method; no series is distinguished by colour alone, and every colour clears a 3:1 contrast
  ratio against the page.
* A method that refuses a budget contributes no point there, so its curve can cover fewer budgets
  than another's. Average ranks across methods with different coverage are not directly
  comparable.
* The small games saturate. At a budget approaching `2^n` every method is near-exact and the
  ranking there is tie-breaking, not comparison; the large games carry the comparison.

## What this run is, and is not

It is a **shared testbed**: one value function family, one budget convention, every registered
approximator, so the methods can be compared against each other on equal terms.

It is **not** a reproduction of any of the three papers. The games here mask features against a
single baseline row and train `XGBRegressor`. The OddSHAP paper, for instance, trains XGBoost
*classifiers* and defines its tabular value function by interventional perturbation estimated from
*50 background instances*, with interventional TreeSHAP as ground truth. Those are different value
functions, so the MSE values here are on a different scale from the ones in the paper
reproductions and should not be compared across the two. Each paper's claims are tested by its own
reproduction under `notebooks/`.
