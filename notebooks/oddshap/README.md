# OddSHAP — paper reproduction

Reproduction of **Fumagalli et al. (2026), "An Odd Estimator for Shapley Values"**
(arXiv:2602.01399) — the paper behind the `OddSHAP` approximator added in
[#522](https://github.com/mmschlk/shapiq/pull/522) and refined in
[#560](https://github.com/mmschlk/shapiq/pull/560).

Everything here reproduces the paper's headline experiments **from the shapiq implementation**, not
from the authors' code.

---

## Headline result

| Claim (paper) | Our reproduction |
|---|---|
| OddSHAP has the best average rank across value functions (Table 1: **1.50**) | **rank 1 on 7 of 8** value functions, **average rank 1.12** |
| Accuracy advantage appears once `m > η·d` (Fig. 2) | reproduced on all 8 value functions |
| Adding odd interactions beats the interaction-free baseline, then over-fits (Fig. 4) | reproduced; same non-monotone shape |

The paper's central claim reproduces. Both the original implementation (#522) and the follow-up
improvements (#560) give **rank 1 on 7/8, average rank 1.125** — i.e. the #560 changes did not alter
the accuracy conclusions (quantified in `oddshap_comparison.ipynb`). The single non-first rank is
`distilbert`, where `RegressionMSR` edges ahead.

The two rank numbers are **not directly comparable**: ours is an average over the six baselines
listed under "What was run" below, which is a smaller pool than the paper's. A smaller pool makes
rank 1 easier to hold, so 1.125 should be read as "the ordering reproduces", not as "we beat the
paper".

---

## Start here — nothing needs to be re-run

The three notebooks are **committed with all outputs**, so they can simply be opened and read.

They are also **fully re-renderable offline in a few minutes**: every number and figure is computed
from the CSVs committed in [`data/`](data), so re-rendering needs **no GPU, no cluster, and no
re-running of the experiments** (and does not require `shapiq` itself to be installed — it is
imported only, if present, to stamp the version into the figures' info banner):

```bash
# from this directory
jupytext --to notebook --set-kernel <your-kernel> --execute nb1_reproduction_ours.py -o oddshap_reproduction.ipynb
```

`data/` holds the results of the actual experiment runs (62 CSVs), which is what makes the
notebooks independently checkable: a reviewer can re-render them and compare, or re-run the
experiments from scratch with the scripts below and regenerate the CSVs.

---

## What is where

| Path | What it is |
|---|---|
| **`oddshap_reproduction.ipynb`** | **Main notebook** — reproduces Fig. 2, Fig. 3, Fig. 4 and Table 1 using our implementation (#522). Start here. |
| `oddshap_reproduction_author.ipynb` | The identical reproduction re-run against the #560 improvements. |
| `oddshap_comparison.ipynb` | #522 vs #560 side by side — quantifies whether the improvements change any conclusion. |
| `nb1_reproduction_ours.py`, `nb2_reproduction_author.py`, `nb3_comparison.py` | The notebook **sources** (jupytext light format). The `.ipynb` files are generated from these; edit these, not the notebooks. |
| `core/` | Shared library: `report.py` (figures/tables), `style.py` (colour-blind-safe palette), `constants.py` (the experiment grid), `harness.py` (games + estimator construction). |
| `data/` | **The experiment results** (committed) — `table1_*`, `fig2_*`, `runtime_*`, `eta_*` per value function and variant. |
| `paper_reference/` | Reference values digitised from the paper's own figures/tables, so our curves can be overlaid against the published ones. |
| `cluster/` | The scripts that produced `data/`: `train_tabular.py`, `train_gpu.py`, `aggregate_gpu.py`. |
| `fleet/` | Helpers used to distribute the GPU runs across rented machines. |
| `experiment_variant_delta.py` | Computes the #522-vs-#560 deltas used by `oddshap_comparison.ipynb`. |

---

## What was run

* **Value functions (8, matching the paper):**
  * tabular — `cancer` (d=30), `realestate` (d=15), `corrgroups60` (d=60),
    `independentlinear60` (d=60), `nhanes` (d=79), `crime` (d=101)
  * vision/language — `vit16` (d=16), `distilbert` (d=14)
* **Baselines (6, from shapiq):** `MSR`, `SVARM`, `PermSamp`, `KernelSHAP`, `kADDSHAP`,
  `RegressionMSR` — plus `OddSHAP`.
* **Repetitions:** **30 instances per (value function, estimator, budget)**; figures report the
  median with an inter-quartile band, tables report median/mean/std.
* **Budgets:** 10 points per tabular value function, from `d+1` up to 20,000.
* **η ablation (Fig. 4):** η ∈ {50, 10, 5, 2} at budget 10,000.
* **Variants:** `v522_merged` (our merged implementation) and `v560_improved` (the follow-up).
* **Ground truth:** exact Shapley values where tractable; the paper's ground-truth method
  otherwise. Each figure carries its exact setup in an info banner, so no figure has to be
  read out of context.

Reproducing the tabular results from scratch (this *does* recompute, and takes hours):

```bash
# from this directory (notebooks/oddshap/)
python cluster/train_tabular.py --vf cancer --variant v522_merged --out data
```

The vision/language value functions were run on rented GPU machines via `fleet/`, and the
per-machine logs merged with `cluster/aggregate_gpu.py`.

---

## Honest limitations

* **The GPU value functions are budget-truncated.** `vit16` and `distilbert` cover 7 budget points
  each, reaching only 1,895 and 1,591 respectively, rather than the paper's full sweep (16,384 /
  20,000). The rented GPU time ran out before the high-budget points finished. Their low-/mid-budget
  behaviour reproduces, and both are low-dimensional (d=16, d=14), where the paper itself notes the
  methods converge — but the high-budget tail for these two is **not** independently confirmed here.
  The η ablation below *is* complete for them at budget 10,000, because it runs OddSHAP alone and so
  costs a fraction of the seven-estimator sweep.
* **`paper_reference/` is digitised from the published figures** (vector PDF), not taken from
  author-provided numbers. It is a visual reference for overlay, accurate to the pixel resolution
  of the source, and should not be read as exact ground truth.
* Figure 4's paper panel omits `realestate`, following the paper, which excludes it as an outlier.

## Conventions

* All figures use a **colour-blind-safe (Okabe–Ito) palette**; no result is encoded by red/green
  alone — every series is distinguishable by marker and line style as well as colour.
* `data/` is committed on purpose: it is the record of the runs, and it is what lets the notebooks
  re-render without a GPU.
