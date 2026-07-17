# PolySHAP: Method Summary

**Paper:** F. Fumagalli, R. Teal Witter, C. Musco, *PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression*, ICLR 2026 ([arXiv:2601.18608](https://arxiv.org/abs/2601.18608)).

## The estimator
KernelSHAP recovers Shapley values (SV) as the solution of a weighted least-squares (WLS) problem in which the game is approximated by a **linear** surrogate fit on sampled coalitions. **PolySHAP generalizes this to a *k-additive polynomial* surrogate**: it fits interaction terms up to a chosen degree `max_order` (the *interaction frontier* `I` in the paper, `explanation_frontier` in the code, i.e. all subsets up to that size), solves the same WLS on randomly sampled coalitions, and reads the SV off the fitted surrogate by splitting each fitted interaction term `φ_T` equally among its `|T|` members (Thm. 4.3). `max_order = 1` reproduces KernelSHAP exactly; higher orders let the surrogate represent genuine feature interactions the linear fit cannot. The `k`-PolySHAP representation coincides with order-`k` Faith-SHAP (Cor. 4.5).

## Convergence and variance
- **Consistency.** Every PolySHAP configuration (any frontier) is a consistent estimator of the *exact* SV: as the budget grows toward full enumeration (`2^n`) the estimate converges to the true value, and at full budget it is exact. KernelSHAP (`max_order = 1`) inherits this too.
- **Variance.** It is a Monte-Carlo estimator; the sampling variance decreases with budget. A higher `max_order` adds frontier terms (parameters `1 + C(n,1) + … + C(n,k)`), so for a *fixed* budget it carries **higher variance** and needs a **larger minimum budget**: a bias/variance / budget trade-off, not free accuracy.
- **Paired sampling (Thm. 5.1).** Paired (antithetic) KernelSHAP returns *exactly* the paired 2-PolySHAP estimate; even-order interactions are captured "for free," giving the first strong theoretical justification for why antithetic sampling works so well.

## `max_order` vs. the game's true interaction order *k*: the key knob
- **`max_order < k` (underfit):** captures only part of the interaction structure, so larger error than optimal at a given budget. `max_order = 1` = plain KernelSHAP.
- **`max_order = k` (match):** captures the game's structure; for a truly *k*-additive game it recovers the **exact** SV once the budget covers the frontier. This is the accuracy sweet spot, and a clear win over KernelSHAP at equal budget.
- **`max_order > k` (overshoot):** the interactions are *already* captured, so there is **no accuracy gain**, only a larger minimum budget and higher variance, meaning **wasted budget, or worse error for the same budget.**

## Integration in shapiq
Implemented as a **single `PolySHAP` approximator** (`shapiq.approximator.PolySHAP`) whose frontier mode is selected by which constructor argument is passed, with no separate variant classes:
- **k-additive (default):** `PolySHAP(n, max_order=k)` builds the full *k*-additive frontier `I_{≤k}` (deterministic, exhaustive; `max_order=1` = KernelSHAP). The recommended default.
- **partial:** `PolySHAP(n, max_terms=ℓ)` builds the budget-controlled frontier `I_ℓ`: whole interaction orders are included from low to high and the single order that does not fit is sampled at random, keeping the noise-robust lower orders complete. For when the full `C(n,k)` frontier is too large for the budget. `max_order` also caps which orders it draws from, so `max_order=n` lets it span all of them.
- **prior:** `PolySHAP(n, prior_frontier=…)` takes the exact interaction terms from the caller. Strong with correct domain knowledge, poor with a wrong prior; not for general use.

`sizes_to_exclude` optionally drops chosen higher orders; `pairing_trick=True` enables paired sampling (which, by Thm. 5.1, already delivers 2-PolySHAP for free). `PolySHAP` is registered as an SV approximator (`SV_APPROXIMATORS`).

## Where it wins / loses
- **Wins:** games with genuine **low-order interactions** (order 2 to 3). A matched `max_order` yields large error reductions over KernelSHAP at the *same* budget, and paired sampling delivers 2-PolySHAP for free.
- **Loses / no benefit:** near-**linear** games (`max_order = 1` is already optimal); **overshooting** `max_order` (pays budget for nothing); and **large `n` with high-order interactions**, where the frontier `C(n,k)` explodes and cheaper samplers scale better.
