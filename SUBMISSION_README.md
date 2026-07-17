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
        *   *Purpose*: Evaluates LeverageSHAP against standard and paired KernelSHAP on real ML datasets (IRIS, California, Diabetes, Breast Cancer, Communities).
        *   *Key Findings*: At small dimensions ($n \le 10$), LeverageSHAP and KernelSHAP with pairing perform similarly. However, at larger dimensions ($n=30$ and $n=99$), LeverageSHAP consistently attains a lower median normalized $\ell_2$ error, matching the paper's claim that the advantage is especially pronounced when the budget $m$ is small relative to $2^n$.
    *   `notebooks/leverageshap/reproduce_leverageshap_soum.ipynb` (**Synthetic SOUM Benchmark**):
        *   *Purpose*: Evaluates the estimator on synthetic Sum of Unanimity Models (SOUM) across player sizes $n \in \{4, 8, 10, 12, 20, 50\}$.
        *   *Key Findings*: Illustrates an honest, minor deviation from the paper's small-$n$ results. On SOUM, LeverageSHAP does not consistently beat paired KernelSHAP. We hypothesize that synthetic unanimity games lack the low-order interaction structures typical of trained models, which might make the leverage-score bounds looser or align exceptionally well with KernelSHAP's subset-size heuristic.
    *   `notebooks/leverageshap/reproduce_figure9_sampling_architecture.ipynb` (**Leverage-Score Sampling vs. Uniform Weights**):
        *   *Purpose*: Isolates the direct benefit of leverage-score sampling by comparing our custom implementation (Algorithm 2) against a uniform-weight KernelSHAP baseline across 25 configurations over 4 datasets.
        *   *Key Findings*: Our custom implementation attains lower absolute errors and successfully resolves the "zig-zag" pattern seen in uniform weight configurations, which typically occurs when budgets run out mid-layer.
    *   `notebooks/leverageshap/reproduce_LeverageSHAP_custom_vs_LeverageSHAPWo2c.ipynb` (**The 2c Threshold Analysis**):
        *   *Purpose*: Studies the exact mechanism of the $2c$ boundary threshold by contrasting the full custom algorithm against a flat-budget variant and a version utilizing the standard `CoalitionSampler`.
        *   *Key Findings*: Isolates the root cause of why the custom sampler outperforms standard stochastic selection. The $2c$ threshold allows deterministic, complete enumeration of low-cardinality size layers ($s=1$ and $s=2$). This reduces sampling variance on these crucial layers (which capture the model's main effects and pairwise interactions) to exactly zero.

*   **Summary & Discussion PDFs (located in the root directory)**:
    *   `leverageshap_summary.pdf` (**Theoretical Method Summary**):
        *   *Content*: Concise mathematical summary of LeverageSHAP. Covers the reformulation of the unconstrained least-squares regression via row-centering and geometric projection, the analytical closed-form derivation of leverage scores ($\ell_s = \binom{n}{s}^{-1}$), the expected budget binary search, and the non-asymptotic $O(n \log n)$ convergence bounds (Theorem 1.1).
    *   `leverageshap_discussion.pdf` (**Empirical Evaluation & Discussion**):
        *   *Content*: Detailed analysis of our benchmarking results. Validates the five paper-derived hypotheses (behavior under small budgets, $m \to 2^n$ convergence, large-$n$ scaling, and high-$\gamma$ plateaus). Includes wall-clock runtime profiling proving LeverageSHAP is $2\times$ to $3\times$ faster than Optimized KernelSHAP due to its SVD-backed solver and uniform size-weighting.

#### 🔹 PolySHAP
*   **Summary Document**: polyshap_summary.md` (A summary of the paper's theory and our integration into shapiq).
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
