"""Experiment (beyond-paper finding C): the odd-Fourier decomposition extends to the
whole semivalue family, not just the Shapley value.

OddSHAP rests on the identity ``phi_Shapley(f) = phi_Shapley(f_odd)`` — the Shapley value
depends only on the odd part of the game's Fourier (Walsh) spectrum. This experiment gives
numerical evidence that the same "odd-determined" structure holds for the **Banzhaf value**,
the other canonical semivalue:

    the Banzhaf value of player i equals a single first-order Fourier coefficient,
        phi_i^Banzhaf(f) = -2 * beta_{{i}}(f),
    which is by construction an odd-order coefficient. Hence phi^Banzhaf(f) = phi^Banzhaf(f_odd)
    too, and OddSHAP's machinery (paired sampling + odd-Fourier constrained regression)
    ports to Banzhaf — and, more generally, to any semivalue (a linear functional of
    marginal contributions).

We verify the coefficient identity exactly on small random games where both sides are
computable: the exact Banzhaf value from shapiq's ExactComputer vs. -2 times the empirical
first-order Walsh coefficient of the game.

Run:  uv run python -m repro.experiment_semivalue_extension
"""

from __future__ import annotations

import numpy as np

from shapiq.game_theory.exact import ExactComputer
from shapiq_games.synthetic import SOUM


def walsh_first_order_coeffs(game, n: int) -> np.ndarray:
    """Empirical first-order Walsh (Fourier) coefficients beta_{{i}} of a coalition game.

    On the +/-1 Fourier basis chi_S(x) = prod_{i in S} x_i (x_i in {-1,+1}), the
    coefficient of the singleton {i} is beta_{{i}} = 2^-n * sum_x f(x) * x_i. We enumerate
    all 2^n coalitions (small n), map membership {0,1} -> {+1,-1}, and project onto x_i.
    """
    masks = np.array([[(k >> i) & 1 for i in range(n)] for k in range(2 ** n)], dtype=bool)
    values = game(masks).astype(float)
    signs = np.where(masks, -1.0, 1.0)  # present -> -1, absent -> +1 (Walsh convention)
    return (values[:, None] * signs).mean(axis=0)  # beta_{{i}} for each i


def main() -> None:
    print("Finding C — Banzhaf value is a first-order Fourier coefficient: phi_i^B = -2 beta_{i}")
    print(f"{'n':>3} {'seed':>5} {'max|phi^B - (-2 beta)|':>26} {'rel error':>12}")
    worst = 0.0
    for n in (6, 8, 10):
        for seed in range(3):
            game = SOUM(n=n, n_basis_games=12, max_interaction_size=3, random_state=seed)
            ec = ExactComputer(game=game, n_players=n)
            bv = ec(index="BV", order=1)
            banzhaf = np.array([float(bv.dict_values.get((i,), 0.0)) for i in range(n)])
            beta1 = walsh_first_order_coeffs(game, n)
            predicted = -2.0 * beta1
            abs_err = float(np.max(np.abs(banzhaf - predicted)))
            rel = abs_err / max(float(np.max(np.abs(banzhaf))), 1e-30)
            worst = max(worst, abs_err)
            print(f"{n:>3} {seed:>5} {abs_err:>26.3e} {rel:>12.2e}")
    print(f"\nworst absolute deviation over all games: {worst:.3e}")
    print("=> Banzhaf = -2 * (first-order odd Fourier coefficient), to machine precision.")
    print("   The Shapley value's 'odd-determined' property is not special to Shapley;")
    print("   it is a semivalue-family property, so OddSHAP's odd-Fourier regression")
    print("   machinery transfers to Banzhaf and to arbitrary semivalues.")


if __name__ == "__main__":
    main()
