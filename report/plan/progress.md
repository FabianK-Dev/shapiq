# Progress

## Stage

**S3/S4** for Chapter 2 — experiments and drafting. The data existed before the writing; the
chapter reports it rather than motivating it.

Chapters 1, 3, 4, 5 are still **S0/S4-skeleton**: structure and configuration tables in place,
prose `TODO`.

## Status by chapter

| Chapter | State | Owner |
|---|---|---|
| 1 Abstract and Introduction | skeleton; abstract to be written last | group |
| **2 Cross-Method Benchmark** | **drafted, both review gates passed** | — |
| 3 OddSHAP | config table filled from the paper; prose TODO | — |
| 4 LeverageSHAP | config table TODO on purpose | teammate |
| 5 PolySHAP | config table TODO on purpose | teammate |

## Chapter 2 — what it argues

The chapter exists to defuse a trap in its own data. The aggregate rank over eleven games puts
LeverageSHAP (3.24) marginally ahead of OddSHAP (3.31), while OddSHAP wins four times as many
cells outright (652 vs 168). Splitting by dimension resolves it: OddSHAP is third on the small
games (4.51) and first by a wide margin on the large ones (1.34), because 24% of small-game
results are already exact and a rank there is a tie-break. On the large games the advantage
switches on at $m > \eta d$ with $\eta = 10$, which is the threshold the OddSHAP paper states.

Both halves match the paper's predictions, on a *different* value function — a cross-check, not a
reproduction.

### Capability-use audit — Chapter 2

- **Required skills:** `paper-orchestration`, `experiment-results-planning`, `writing-chapters`,
  `evidence-driven-writing`.
- **Skills actually used:** all four. `figures-python` not needed — the figures already existed
  from the benchmark run; two were copied into `report/figures/` rather than redrawn.
- **Inputs consumed:** `benchmark/results/lmu_full_sweep_20260717/results.csv` (19,800 cells) and
  its README; `benchmark/performance.py`; arXiv:2602.01399 §5 (fetched, read for its baseline set
  and its budget-threshold claim); `plots_paper_subset/` figures.
- **Inputs not used and why:** `plots/` (all fifteen methods) — unreadable at that density, and
  most of those methods are not in any paper's comparison; the eight-method subset is used
  instead. `notebooks/oddshap/` — belongs to Chapter 3, different value function.
- **Artifacts produced:** `content/02-benchmark.tex` (~1,265 words of prose, two figures, one
  configuration table); `figures/bench-communities-n101.png`; `figures/bench-soum-n10.png`;
  `plan/project-overview.md`; `plan/task-packets/02-benchmark.md`.
- **Verification run:** every one of the 20 numbers in the chapter re-derived from the CSV
  independently of the text — all matched. `latexmk -pdf main.tex` exits 0; 18 pages; zero
  unresolved references. Spec checks against the packet's rejection list: no placeholder text, no
  bullet lists, aggregate rank not presented as the verdict, cross-check stated explicitly,
  cross-chapter MSE comparison explicitly ruled out.
- **Remaining risk:**
  1. The small/large split at $n \le 12$ vs $n \ge 60$ is a gap in the grid, not a principled
     threshold; there is no game between 12 and 60 to test where the inversion happens.
  2. Ten predictions per game make the quartiles coarse. Stated in the chapter.
  3. The rank table promised in the packet was folded into prose. Ranks are given in the text; a
     separate table would repeat them.
  4. The claim that the gap widens with budget (LeverageSHAP 2.05 → 2.65) is read off mean ranks,
     not tested for significance.
