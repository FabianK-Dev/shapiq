# Cross-Method Performance Benchmark

Self-contained runner that benchmarks every SV approximator registered in
`shapiq.approximator.SV_APPROXIMATORS` against `ExactComputer` ground truth.
Drops into any feature branch without touching anyone else's source — pull
this branch in, run the CLI, get performance curves.

## How to merge into your branch

```bash
# from your approximator feature branch:
git fetch origin
git merge origin/wu/conformance-test
```

The merge adds one top-level path only:

- `benchmark/performance.py` — performance sweep CLI.
- `benchmark/_discovery.py` — resolves the approximator list at run time.
- `benchmark/README.md` — this file.

No other files are modified, so the merge is clean by construction.

## Make sure your approximator is discovered

The benchmark sources the list of approximators **dynamically** from:

```python
shapiq.approximator.SV_APPROXIMATORS  # list of approximator classes
```

If your new method is already registered there, you're done. If not,
add one line to `src/shapiq/approximator/__init__.py`:

```python
SV_APPROXIMATORS: list[Approximator.__class__] = [
    OwenSamplingSV,
    StratifiedSamplingSV,
    SVARM,
    # ... existing ...
    YourNewApproximator,  # <-- here
]
```

That's the only change required.

## Probe interface compatibility

Before running a full sweep, you can check that every method is
constructible in SV mode:

```bash
uv run python -m benchmark.performance --check
```

Sample output:

```
Method                    Registered  Constructible  Notes
--------------------------------------------------------------------------------
OwenSamplingSV            yes         yes            OK (OwenSamplingSV)
KernelSHAP                yes         yes            OK (KernelSHAP)
SPEX                      yes         yes            OK (SPEX)
LeverageSHAP              no          -              not exported by shapiq.approximator
OddSHAP                   yes         yes            OK (OddSHAP)
```

## Run the benchmark

```bash
# Full default sweep — every registered SV method, 4 budgets, 3 seeds, with plots.
# Games: SOUM(n in 6/8/10) plus the small ML games (IRIS, California, Diabetes, Adult).
uv run python -m benchmark.performance --plot

# Restrict to one method — output goes to oddshap_bench_<ts>/
uv run python -m benchmark.performance --methods OddSHAP --plot

# Include the large ML games (n = 60/79/101) — see "Budgets and game size" below
uv run python -m benchmark.performance --budget-mults 2,5,10,25 --plot

# Quick smoke run for development
uv run python -m benchmark.performance \
    --n 6 --budgets 0.25,1.0 --seeds 0
```

### Budgets and game size

`--budgets` are **fractions of the 2^n coalition space**. That only expresses a runnable
budget while `n` is small: on `Independent(n=60)` even 5% is 5.8e16 evaluations, so the
sweep would never return. The default run therefore **skips games above `n=20` and says so
on stderr** rather than hanging.

To benchmark the large ML games (`Independent`/`Correlated` at n=60, `NHANES` at n=79,
`Communities` at n=101), use `--budget-mults`, which sizes the budget as `mult * n` — the
convention the approximator papers use — and applies to every game:

```bash
uv run python -m benchmark.performance --budget-mults 2,5,10,25 --plot
```

Default arguments:

| Flag            | Default                                                       |
|-----------------|---------------------------------------------------------------|
| `--methods`     | `all` (every registered SV approximator + the 3 project ones) |
| `--n`           | `6,8,10` (SOUM player counts; the ML games are always included) |
| `--budgets`     | `0.05,0.25,0.5,1.0` (fractions of 2^n; games with n > 20 are skipped) |
| `--budget-mults`| unset (if given, budget = `mult * n` and every game is included) |
| `--seeds`       | `0,42,1337`                                                   |
| `--name`        | derived (`<method>_bench` for a single method, else `sv_sweep`) |
| `--output-root` | `benchmark/results`                                           |
| `--plot`        | off                                                           |

## Output layout

Each run creates a timestamped folder under `--output-root`:

```
benchmark/results/
├── sv_sweep_20260518_201500/                # full multi-method run
│   ├── results.csv
│   └── plots/                               # only when --plot is passed
│       ├── MSE_SOUM_n6.png
│       ├── MAE_SOUM_n6.png
│       ├── SSE_SOUM_n6.png
│       ├── SAE_SOUM_n6.png
│       ├── Precision_at_5_SOUM_n6.png
│       ├── Precision_at_10_SOUM_n6.png
│       ├── KendallTau_SOUM_n6.png
│       ├── runtime_SOUM_n6.png
│       └── … (same set per game)
└── oddshap_bench_20260518_204230/           # single-method run
    ├── results.csv
    └── plots/…
```

Pass `--name <custom>` to override the folder name.

## CSV format (long-form)

One row per `(method, game, n, budget, seed, metric, value)`:

```csv
method,game,n,budget,seed,metric,value,runtime_seconds,status
KernelSHAP,SOUM(n=6),6,16,0,MSE,0.0124,0.012,ok
KernelSHAP,SOUM(n=6),6,16,0,MAE,0.083,0.012,ok
KernelSHAP,SOUM(n=6),6,16,0,Precision@5,0.8,0.012,ok
KernelSHAP,SOUM(n=6),6,16,0,KendallTau,0.73,0.012,ok
SPEX,SOUM(n=6),6,16,0,_,,0.001,"skipped:refused_regime(ValueError)"
```

`status` is `"ok"` for successful cells. Otherwise it carries the skip
reason (`"skipped:refused_regime(ValueError)"` when a sparse method
explicitly rejects a budget regime, `"skipped:not_registered"` when the
class is not exported by `shapiq.approximator`, etc.). Skipped cells are
retained in the CSV for auditing but excluded from the plots.

## Metrics produced

| Metric         | Definition                                                              |
|----------------|-------------------------------------------------------------------------|
| `MSE`          | Mean squared error vs exact Shapley values                              |
| `MAE`          | Mean absolute error                                                     |
| `SSE`          | Sum squared error                                                       |
| `SAE`          | Sum absolute error                                                      |
| `Precision@5`  | Overlap of top-5 \|value\| features between estimate and ground truth   |
| `Precision@10` | Overlap of top-10 \|value\| features                                    |
| `KendallTau`   | Rank correlation of feature attributions                                |

`MSE`, `MAE`, `SSE`, `SAE` operate on the full SV vector (including the
empty-coalition baseline). `Precision@k` and `KendallTau` operate on the
singleton portion only.

Plots additionally include a `runtime_<game>.png` showing wall-clock
runtime versus budget per method, on log-log axes.

## Plot conventions

- One curve per method per `(game, metric)`.
- X-axis = budget on log scale.
- Y-axis: `MSE`/`MAE`/`SSE`/`SAE` use `symlog` so machine-precision zeros
  do not break the scale.
- Shaded bands show ±1 σ across seeds (only visible when the std is
  non-zero, i.e. when at least two seeds were run).

## Adding a new metric or a new game

Edit `METRIC_FUNCTIONS` or `default_game_specs` in
`benchmark/performance.py`. Both are flat dictionaries / lists and the
CSV columns / plot files derive from them at run time. No other change
is required.

## Notes on multi-index approximators

`SPEX`, `ProxySPEX`, `ProxySHAP`, `MSRBiased`, `kADDSHAP` default to
`index="FBII"` with `max_order=n`. The runner tries the explicit SV
signature first (`Approx(n=n, index="SV", max_order=1, random_state=...)`)
and falls back to the minimal signature for SV-only methods. If neither
works, the cell is skipped with status `"skipped:incompatible_constructor"`.

Some sparse methods (e.g. `SPEX`) raise `ValueError("Insufficient
budget…")` below their internal minimum — captured as
`"skipped:refused_regime(ValueError)"` and excluded from plots but kept
in the CSV.
