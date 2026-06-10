# Full-scale OddSHAP reproduction results (LMU CIP cluster)

Aggregated results of the paper-standard reproduction runs (Fumagalli et al. 2026,
arXiv:2602.01399), generated on the LMU CIP Slurm cluster with the same methodology as
the gallery scripts in `examples/approximators/` (which default to reduced instance
counts for fast builds; these CSVs are the full-scale numbers).

Common configuration: `random_state=40`, 50 background instances, exact interventional
Shapley ground truth from shapiq's `InterventionalTreeExplainer` (`index="SV"`),
Fourier-screened OddSHAP (post `oddshap_approximator` fix), XGBoost value-function
models per paper Section 5.

| File | Contents |
|---|---|
| `table1_n30_all_classifier.csv` | Table-1 reproduction, N=30 instances/VF, budget `100*d`, all six tabular VFs as XGBoost **classifiers** (paper Section-5 text). OddSHAP rank-1 on 6/6. |
| `table1_n30_estate_crime_regressor.csv` | Same, with the **regressor** reading of the continuous Estate/Crime targets (matches the paper's Table-3 error magnitudes). OddSHAP rank-1 on 6/6. |
| `eta_ablation_n30_budget10000.csv` | Figure-4 interaction-sparsity ablation at the paper's fixed budget of 10,000; `eta` in {50,10,5,2} plus the interaction-free baseline; MSE ratio column normalised by that baseline. |
| `fig2_budget_curves_n10_classifier.csv` | Figure-2 budget curves: median MSE over N=10 instances at 10 log-spaced budgets `d+1 .. min(2^d, 20000)`. Budgets an estimator refuses (OddSHAP below `d*eta`) are absent by design. |
| `distilbert_n30.csv` | Table-1 language column: DistilBERT sentiment (`lvwerra/distilbert-imdb`, d~10-15), N=30 reviews, exact ground truth via `ExactComputer`, budget `100*d`. |

The two Estate/Crime configurations exist because the paper is internally ambiguous
(Section 5 says "XGBoost classifiers" for all tabular value functions, while the
Table-3 magnitudes for the continuous targets are only reproduced by a regressor).
The ranking conclusion — OddSHAP rank-1 on all six tabular value functions — holds
under either reading.
