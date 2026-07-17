# Project Submission: New Shapley Value Approximators for shapiq

This repository contains our team's implementation, validation, reproduction studies, and benchmarking for three recent state-of-the-art Shapley value approximators: **LeverageSHAP**, **PolySHAP**, and **OddSHAP**.

> **⚠️ NOTE FOR REVIEWERS (Task 1 & Task 2)**
> The core approximator implementations under `src/` and their respective unit tests under `tests/` correspond to Task 1 and Task 2. While they are merged into this branch to ensure the test suite compiles and runs correctly, **`src/` and `tests/` do not need to be reviewed additionally.**

---

## 📂 Repository Layout & Deliverables

Below is a detailed index of where to find the deliverables for our paper reproductions (Task 3) and cross-method benchmarking (Task 4).

### 1. Paper Reproductions (Task 3)

#### 🔹 LeverageSHAP
Our LeverageSHAP reproduction contains four active notebooks mapping to the paper's key experimental setups, alongside two comprehensive PDF deliverables located directly in the root directory.

*   **Reproduction Notebooks (located in `notebooks/leverageshap/`)**:
    *   `notebooks/leverageshap/reproduce_leverageshap_ml.ipynb` (**XGBoost Models on Real Data**):
        *   *Purpose*: Evaluates LeverageSHAP against standard and paired KernelSHAP on real ML datasets ($n \le 99$).
        *   *Key Findings*: Performance gains over paired KernelSHAP scale with dimension $n$ and are most substantial in the highly constrained budget regime.
    *   `notebooks/leverageshap/reproduce_leverageshap_soum.ipynb` (**Synthetic SOUM Benchmark**):
        *   *Purpose*: Evaluates the estimator on synthetic Sum of Unanimity Models (SOUM) across player sizes $n \le 50$.
        *   *Key Findings*: Documents a minor, structured deviation where LeverageSHAP and paired KernelSHAP perform on par on highly structureless synthetic games.
    *   `notebooks/leverageshap/reproduce_figure9_sampling_architecture.ipynb` (**Leverage-Score Sampling vs. Uniform Weights**):
        *   *Purpose*: Isolates the benefit of leverage-score sampling against a uniform-weight KernelSHAP baseline across 25 configurations.
        *   *Key Findings*: Validates consistently lower errors and smoother convergence trends for leverage-score sampling.
    *   `notebooks/leverageshap/reproduce_LeverageSHAP_custom_vs_LeverageSHAPWo2c.ipynb` (**The 2c Threshold Analysis**):
        *   *Purpose*: Isolates the mechanism of the $2c$ boundary threshold against a flat-budget variant and the standard `CoalitionSampler`.
        *   *Key Findings*: Proves that the $2c$ threshold improves accuracy by allowing deterministic, zero-variance enumeration of low-cardinality layers ($s=1, 2$).

*   **Summary & Discussion PDFs (located in the root directory)**:
    *   `leverageshap_summary.pdf` (**Theoretical Method Summary**):
        *   *Content*: Mathematical summary of row-centering, projection matrices for unconstrained WLS, closed-form leverage scores, and the $O(n \log n)$ convergence proof.
    *   `leverageshap_discussion.pdf` (**Empirical Evaluation & Discussion**):
        *   *Content*: Empirical evaluation validating five core paper hypotheses against LMU CIP benchmark results, alongside wall-clock runtime profiling of SVD-backed least-squares.

#### 🔹 PolySHAP
*   **Summary Document**: `polyshap_summary.md` (A summary of the paper's theory and our integration into shapiq).
*   **Reproduction Notebooks**:
    *   `notebooks/polyshap/polyshap_reproduction.ipynb`: Side-by-side comparison of our integrated implementation against the original published ICLR 2026 panels (ResNet18 and ViT16).
    *   `notebooks/polyshap/polyshap_maxorder_vs_k.ipynb`: Evaluates performance when the chosen `max_order` under-fits, matches, or over-fits a game with a known interaction order.
    *   `notebooks/polyshap/polyshap_true_order.ipynb`: Evaluates the late, non-linear convergence of under-parameterized models near full enumeration.

#### 🔹 OddSHAP
*   **Summary Document**: `oddshap_summary.md` (A detailed summary of the paper's theory and our implementation deviations).
*   **Reproduction Notebooks**:
    *   `notebooks/oddshap/oddshap_reproduction.ipynb`: Reproduces the core figures and Table 1 using our merged implementation (PR #522).
    *   `notebooks/oddshap/oddshap_reproduction_author.ipynb`: Runs the same suite against the author's follow-up changes (PR #560).
    *   `notebooks/oddshap/oddshap_comparison.ipynb`: A direct comparison proving both versions yield identical results in the paper-scale budget regime.
*   **Harness & Raw Data**: Located in `notebooks/oddshap/data/` (committed results allowing offline rendering) and `notebooks/oddshap/core/`.

---

### 2. Cross-Method Benchmark & Report (Task 4)
We implemented a command-line benchmarking suite to evaluate the new approximators against all existing `shapiq` estimators under uniform conditions.

*   **Benchmarking CLI**: `benchmark/performance.py` (A self-contained runner evaluating error metrics vs. budgets).
*   **Scale of Evaluation**: Runs are documented in `benchmark/results/lmu_full_sweep_20260717/`, comprising a multi-seed evaluation on the LMU CIP cluster across 11 games and 14 estimators.
*   **Benchmarking Plots**: Located in `benchmark/results/lmu_full_sweep_20260717/plots/`.
*   **Final Report**: The LaTeX source files for our final project report are located in the `report/` directory:
    *   Main file: `report/new-shapley-value-approximators.tex`
    *   Compiled figures: `report/figures/`

---

Additionally, you can view all individual changes and new files compared to the main branch on `mmschlk/shapiq` here: https://github.com/mmschlk/shapiq/compare/main...FabianK-Dev:shapiq:main
