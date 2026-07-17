# Focused figures — the group's approximators against the OddSHAP paper's baselines

The same run as `../results.csv`, drawn with eight methods instead of fifteen. The full set is in
`../plots/`; nothing here is a different measurement, only a different selection.

Fifteen curves on one axis is unreadable, and most of those methods are not what any of the three
papers compare against. These figures keep only:

**The three the group contributed**

* `OddSHAP`, `LeverageSHAP`, `PolySHAP`

**The baselines the OddSHAP paper compares against**, in its own words (§5):

> "We compare OddSHAP-η with η=10 against **Permutation Sampling** (Castro et al., 2009), **SVARM**
> (Kolpaczki et al., 2024), **MSR** (Witter et al., 2025), which is equivalent to **Unbiased
> KernelSHAP** (Covert & Lee, 2021; Fumagalli et al., 2023), **FourierSHAP** (Gorji et al., 2025),
> and **LeverageSHAP** (Musco & Witter, 2025) [...] We further compare against the proxy-based
> methods **RegressionMSR** (Witter et al., 2025) and **Proxy (LGBM)**, which corresponds to
> RegressionMSR without adjustment, or equivalently **ProxySPEX** (Butler et al., 2025)."

| Paper's name | Here |
|---|---|
| Permutation Sampling | `PermutationSamplingSV` |
| SVARM | `SVARM` |
| MSR / Unbiased KernelSHAP | `UnbiasedKernelSHAP` |
| LeverageSHAP | `LeverageSHAP` |
| RegressionMSR | `RegressionMSR` |
| Proxy (LGBM) / ProxySPEX | `ProxySPEX` |

**Not here, because shapiq does not have them:** `FourierSHAP` (Gorji et al., 2025), and the
fixed-budget `FFD-RD` and `FFD-RD (corrected)` variants (Zhou et al., 2025). The paper compares
against those too, so this selection is the paper's baseline set *intersected with* what shapiq
registers, not the paper's set in full.

**Dropped from the full set:** `KernelSHAP`, `OptimizedKernelSHAP`, `OwenSamplingSV`,
`StratifiedSamplingSV`, `kADDSHAP`, `SPEX`, `ShaplEIG`. They are in shapiq and in `../plots/`, but
the OddSHAP paper does not use them as baselines.

## Reading these

Same conventions as the full set: each point is the median over five seeds with an interquartile
band; every series has its own colour, marker and dash pattern, and none is identified by colour
alone.

These are still the benchmark's value function — features masked against a single baseline row,
`XGBRegressor` — not the OddSHAP paper's, which uses XGBoost classifiers and interventional
perturbation over 50 background instances. The selection of methods follows the paper; the game
does not. See `../README.md`.

## Regenerating

Filter `../results.csv` to the eight methods above and pass the rows to
`benchmark.performance.plot_all_figures(results, out_dir, style_name="default")`.
