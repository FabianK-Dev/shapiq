# Project overview

**Deliverable.** The Group G report for the LMU practical *New Shapley Value Approximators*.
Source in `report/`, built with `latexmk -pdf main.tex`.

**Paper type.** Course report, not a thesis and not a conference paper. Five chapters, fixed by
the group:

1. Abstract and Introduction
2. Cross-Method Benchmark
3. OddSHAP — Summary, Reproduction
4. LeverageSHAP — Summary, Reproduction
5. PolySHAP — Summary, Reproduction

**What the report has to answer.** Two different questions, with two different setups:

* *Do the papers reproduce?* Chapters 3–5. Each uses **its own paper's value function**, because
  that is the only setup in which that paper's claim can be tested.
* *How do the methods compare?* Chapter 2. One **shared** value function for every method, which
  then belongs to no paper in particular.

Absolute errors are therefore not comparable across the two, and the introduction says so.

**Evidence base.** Everything is committed; no number in the report may come from memory.

| Source | What it backs |
|---|---|
| `benchmark/results/lmu_full_sweep_20260717/results.csv` | Chapter 2. 19,800 cells, 143,980 metric rows. Its `README.md` records the command, code revision and machine. |
| `notebooks/oddshap/` | Chapter 3. 62 CSVs, three rendered notebooks. |
| LeverageSHAP notebooks | Chapter 4. Owner: teammate. |
| `notebooks/polyshap/` | Chapter 5. Owner: teammate. |
| arXiv:2602.01399 (OddSHAP) | Chapter 3 configuration, verbatim from §5. |

**Ownership.** Chapters 4 and 5 belong to the teammates who wrote those approximators. Their
configuration tables are deliberately left `TODO`: guessing another author's experimental setup is
how a wrong configuration ends up in a report.

**Constraints.**

* No invented citations. `references.bib` is extracted from `docs/source/references.bib`, so keys
  match the codebase.
* No claim without a committed artefact behind it.
* Do not present our average rank of 1.125 against the OddSHAP paper's 1.50 as an improvement:
  ours is over a smaller baseline pool. It reproduces the ordering, not the number.
