# Cross-method sweep — 2026-07-17 (LMU CIP cluster)

Every SV approximator registered in `shapiq.approximator.SV_APPROXIMATORS`, run on a shared set
of games over a budget grid, five seeds per point. This is the run the cross-method discussion
(task 4) is written against.

## How it was generated

The sweep is `benchmark/performance.py`, nothing else. It is equivalent to this single command:

```bash
python -m benchmark.performance --budget-mults 1,2,5,10,25,50,100 --seeds 0,42,1337,7,99 --plot
```

It was run as five cluster jobs instead, one per seed, purely to cut wall-clock time. Seeds are
independent — a seed fixes `train_test_split`, the model, and the approximator — so running them
in five processes and concatenating the CSVs yields the same cells as one process running all
five. Per job:

```bash
python -m benchmark.performance \
    --budget-mults 1,2,5,10,25,50,100 \
    --seeds "$SEED" \
    --name "full_seed${SEED}" \
    --output-root ~/bench_full/results
```

`--plot` was not passed to the jobs: five single-seed plot sets would each show a band over one
seed. The figures here were produced afterwards from the merged `results.csv` via
`plot_all_figures(results, out_dir, style_name="default")`, which is the same function `--plot`
calls.

| | |
|---|---|
| Code | `582a3ec4` (branch `submission`) |
| Where | LMU CIP cluster, `krater20`, Slurm job `200659`, array 0-4 |
| Nodes | 14 cores per task |
| Wall-clock | 12-13 min per task, run in parallel |
| Date | 2026-07-17 |

## Why `--budget-mults` and not the default `--budgets`

`--budgets` are fractions of the 2^n coalition space. That only expresses a runnable budget while
n is small: at n=60 even 5% is 5.8e16 evaluations. It also saturates — at 100% of 2^n every method
is exact, so the ranks there are ties, not a comparison.

`--budget-mults` sizes the budget as `mult * n`, which is the convention the approximator papers
use, applies to every game including the large ones, and puts the large games in the
under-determined regime where the methods actually differ. On `Communities(n=101)` the grid spans
101 to 10,100 evaluations, i.e. `1*d` to `100*d`.

## What ran

* **Games (11).** `IRIS(n=4)`, `SOUM(n=6)`, `California(n=8)`, `SOUM(n=8)`, `Diabetes(n=10)`,
  `SOUM(n=10)`, `Adult(n=12)`, `Correlated(n=60)`, `Independent(n=60)`, `NHANES(n=79)`,
  `Communities(n=101)`. The ML games follow the eight-dataset grid of Musco & Witter (2024).
* **Methods (15 registered, 14 ran).** `ShaplEIG` is skipped everywhere: it needs the `shapleig`
  extra (torch, gpytorch, botorch), which is not installed. That is an environment gap, not a
  defect — the rows record it as `skipped:incompatible_constructor(ImportError)`.
* **Budgets.** `mult * n` for `mult` in 1, 2, 5, 10, 25, 50, 100.
* **Seeds.** 0, 42, 1337, 7, 99.
* **Ground truth.** Games up to ten players enumerate the game exactly (`ExactComputer`). Larger
  games use interventional TreeSHAP against the same single baseline row the game masks with, so
  the target is the Shapley value of the game the approximators are actually given.

## Contents

| File | What |
|---|---|
| `results.csv` | 41,720 metric rows over 5,775 cells (method x game x budget x seed), long format |
| `seed_*.csv` | the five per-seed CSVs the jobs wrote, before merging |
| `plots/` | 99 figures: nine metrics across the eleven games |

Status counts: 41,080 `ok`, 385 `skipped:incompatible_constructor` (all `ShaplEIG`), 255
`skipped:refused_regime` (methods that decline a budget below their minimum — OddSHAP, SPEX and
ProxySPEX do this by design rather than silently returning a degraded estimate).

## Reading the figures

* Each point is the **median over the five seeds**, and the band is the **interquartile range**.
  With five seeds the quartiles are a coarse envelope; read them as spread, not as a confidence
  interval.
* Every method has its own colour **and** marker **and** dash pattern, from a colour-blind-safe
  palette. No series is distinguished by colour alone.
* A method that refuses a budget contributes no point there, so its curve can cover fewer budgets
  than another's. Average ranks across methods with different coverage are not directly
  comparable.

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
