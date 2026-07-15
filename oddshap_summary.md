# OddSHAP Paper Summary

Fumagalli et al. (2026) :cite:t:`Fumagalli.2026`

## The core idea

Every set function $f: 2^{[d]} \to \mathbb{R}$ splits into an odd part and an even part, $f = f_{odd} + f_
{even}$, where $f_{odd}(S) = \frac{f(S) - f(S^c)}{2}$ and $f_{even}(S) = \frac{f(S) + f(S^c)}{2}$. The paper's key
observation (Observation 3.1) is that the Shapley value only depends on $f_{odd}$: $\phi_i(f) = \phi_i(f_{odd})$ for
every player $i$. That's because marginal contributions of complementary coalitions cancel out the even part when
summing over all subsets.

This explains why paired sampling (evaluating $S$ and $S^c$ together) helps so much in practice. The authors show (
Theorem 3.2) that under paired samples, a regression objective over a function class closed under odd/even decomposition
splits into two independent problems, one for the odd part and one for the even part. Since only the odd part
matters for $\phi_i$, paired sampling is implicitly filtering out irrelevant even structure.

This also motivates the OddSHAP estimator itself: since we only need to recover $f_{odd}$, we
should not waste budget fitting even-order terms at all.

## Why the Fourier basis

The usual regression estimators (KernelSHAP, LeverageSHAP, PolySHAP) work in the unanimity/Möbius basis, but those basis
functions mix odd and even components. The Fourier (Walsh) basis $\chi_T(S) = (-1)
^{|S \cap T|}$ doesn't have this problem: $\chi_T$ is an odd function exactly when $|T|$ is odd. So, the authors
restrict a
Fourier regression to odd-cardinality terms only. They also show (Theorem 3.5) that this constrained Fourier regression
is still consistent. This means it recovers the Shapley values via the same projection used for KernelSHAP and PolySHAP.

## The estimator: sampling and selection

Algorithm 1 (OddSHAP) has three steps:

1. **Paired sampling:** Coalitions are drawn without replacement, uniformly over coalition sizes, and every sampled $S$
   is paired with its complement $S^c$.
2. **Interaction screening:** A gradient-boosted tree (GBT) is fit to the sampled coalitions, converted to its
   Fourier spectrum, and the odd-cardinality terms with the largest coefficients are kept as candidates. The candidate
   budget is $\lceil m/\eta \rceil$, where $\eta$ ("interaction_factor") trades off support size against regression
   budget; the paper's default is $\eta = 10$.
3. **Sparse odd regression:** A weighted least-squares problem is solved only over singletons plus the selected
   higher-order odd terms, with the empty/full coalition values used as exact boundary constraints (efficiency is
   enforced, not "softly" via huge weights). Shapley values are then read off in closed form as $\phi_i =
   -2\sum_{T \ni i, |T| \text{odd}} \beta_T / |T|$.

If the budget is too small to even fit the linear terms ($m < d\eta$), the paper falls back to reading Shapley values
directly off the GBT via TreeSHAP.

## Experiments

The paper benchmarks 8 value functions (DistilBERT and ViT-16 from `shapiq`, plus tabular XGBoost/synthetic games up to
$d=101$) against Permutation Sampling, SVARM, MSR, LeverageSHAP, PolySHAP, and RegressionMSR, measuring MSE vs. budget
$m$ (Figure 2) and vs. wall-clock runtime under simulated evaluation costs (Figure 3, §5.2). OddSHAP gets the best
average rank overall (1.50, Table 1) and clearly wins once $m > \eta d$; at very low budgets it degrades to
the GBT/TreeSHAP fallback and performs comparably to RegressionMSR/LeverageSHAP.
An ablation (Figure 4, §5.3) shows adding up to ~1,000 odd interactions gives a 6×–62× MSE reduction over the
interaction-free baseline (LeverageSHAP), but too many interactions eventually overfit and reverse the gain. This is why
the candidate budget is capped at $\lceil m/\eta \rceil$ rather than enumerated.

## Deviations in this implementation

Our implementation follows Algorithm 1 closely but diverges in the low-budget fallback:

- **No TreeSHAP fallback.** The paper falls back to TreeSHAP on the fitted GBT whenever $m < d\eta$. We deliberately
  don't do this. An under-budgeted call to our `OddSHAP` never silently returns a different estimator's values.
  Instead, we lower the minimum usable budget to `interaction_factor` (the paper's $\eta$) and raise `ValueError` below
  that, unless the budget already covers the full coalition space ($m \geq 2^d$).

Therefore, our numbers may diverge from the paper specifically in the low-budget case.
