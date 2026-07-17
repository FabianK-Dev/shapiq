# Task packet — Chapter 2, Cross-Method Benchmark

**Scope.** Write the prose of Chapter 2: the paragraph after the configuration table (Setup), and
the whole of *Results and Discussion*, including the threats-to-validity paragraph. The
configuration table itself already exists and is verified.

**Files to read.**
* `benchmark/results/lmu_full_sweep_20260717/README.md` — provenance, verified against the CSV.
* `benchmark/results/lmu_full_sweep_20260717/results.csv` — the only source of numbers.
* `benchmark/performance.py` — what the sweep actually does.
* arXiv:2602.01399 §5 — what the OddSHAP paper predicts, for the "based on what the papers
  predict" part of the deliverable.

**Files allowed to edit.** `report/content/02-benchmark.tex` only.

**Required skills.** `experiment-results-planning` (what the chapter must report),
`writing-chapters` (prose), `evidence-driven-writing` (every claim tied to an artefact).

**Evidence — computed from the CSV, all verified.**

| Fact | Value |
|---|---|
| Aggregate rank, all 11 games | LeverageSHAP 3.24 · **OddSHAP 3.31** · PolySHAP 4.50 · OptimizedKernelSHAP 4.69 · RegressionMSR 5.46 |
| Outright wins, all 11 games | **OddSHAP 652** · LeverageSHAP 168 · OptimizedKernelSHAP 110 · PolySHAP 99 |
| Small games (n≤12), 840 cells | LeverageSHAP 3.55 · PolySHAP 3.68 · **OddSHAP 4.51** |
| Large games (n≥60), 480 cells | **OddSHAP 1.34** · LeverageSHAP 2.69 · OptimizedKernelSHAP 4.58 |
| Saturation | small: **24.0%** of results already exact (MSE<1e-10); large: **1.0%** |
| OddSHAP vs budget, large games | 1d: 3.40 → 2d: 1.35 → 5d: 1.30 → **12d–160d: 1.00–1.07** |
| Runner-up at 12d+ | LeverageSHAP, 2.05 → 2.65 (drifts *away* as budget grows) |
| Refusals | OddSHAP 50 (all at 1d/2d on IRIS, SOUM6/8, California — i.e. budget < η=10); SPEX 680; ProxySPEX 10 |
| ShaplEIG | 1320 cells skipped, missing optional extra |
| OddSHAP coverage | 1270/1320 overall; **480/480 on the large games** |

**The argument the chapter must make.** In this order:

1. The aggregate rank over all eleven games is **regime-confounded and should not be read as a
   verdict**: LeverageSHAP leads it (3.24 vs 3.31) while OddSHAP takes four times as many outright
   wins (652 vs 168). The average is dominated by the small games.
2. Split by dimension and the ordering **inverts**: OddSHAP is third on n≤12 (4.51) and first by a
   wide margin on n≥60 (1.34).
3. That inversion is not noise, it is **saturation**: 24% of small-game results are already exact,
   where a rank is tie-breaking; only 1% of large-game results are. The large games carry the
   comparison.
4. On the large games the advantage **switches on at a predicted budget**: rank 3.40 at 1d, ≈1.0
   from 12d. With η=10 that is `m > η·d`, which is what the OddSHAP paper states.
5. Both halves match what the papers predict — the paper itself says OddSHAP matches
   RegressionMSR on low-dimensional value functions and outperforms on moderate-to-high ones. Say
   this is a *cross-check on a different value function*, not a reproduction.
6. Refusals: OddSHAP's 50 are all below its documented minimum budget on tiny games, none on the
   large games. Average ranks across methods with different coverage are not directly comparable —
   SPEX ran in far fewer cells.

**Rejection checks.**
* No number that is not in the table above or recomputable from the CSV.
* Do not claim this reproduces or refutes any paper: different value function.
* Do not present the aggregate rank as the headline; that is the trap the chapter exists to defuse.
* No bullet lists in the body. Prose.
* Do not compare MSE magnitudes with Chapters 3–5.

**Required artifacts.** Prose in `02-benchmark.tex`; one figure reference; one rank table.

**Validation.** `latexmk -pdf main.tex` from `report/`; every cited number re-derived from the CSV.
