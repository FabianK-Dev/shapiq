"""Reproduce the OddSHAP paper's DistilBERT value function (Table 1, language column).

This standalone script reproduces the neural ``DistilBERT`` value function from the
OddSHAP paper (Fumagalli et al. 2026, arXiv:2602.01399). It is kept separate from the
tabular reproduction notebook/gallery example because the DistilBERT games require a
huggingface model (``lvwerra/distilbert-imdb``) and benefit from a GPU — the tabular
notebook cannot generate these results.

For each short IMDB-style review, the value function is the model's sentiment score; the
number of players ``d`` equals the number of tokens (``d`` ~ 10-14). Because ``d`` is
small, the **exact** Shapley values are computed with shapiq's ``ExactComputer`` and used
as ground truth. Each Shapley-value approximator is run at the paper's budget
``m = 100 * d`` and scored by mean-squared error against the exact values.

Run a quick gallery build on CPU::

    python plot_oddshap_distilbert_reproduction.py

Reproduce the paper's Table-1 figures (30 reviews, GPU)::

    N_INSTANCES = 30  # edit below
    DEVICE = "cuda"   # edit below
"""

from __future__ import annotations

import numpy as np

from shapiq import ExactComputer
from shapiq.approximator import (
    SVARM,
    KernelSHAP,
    OddSHAP,
    PermutationSamplingSV,
    RegressionMSR,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq_games.benchmark.local_xai import SentimentAnalysis

# ---------------------------------------------------------------------------
# Configuration  (set N_INSTANCES = 30, DEVICE = "cuda" for the paper's run)
# ---------------------------------------------------------------------------
N_INSTANCES = 3
DEVICE: str | None = None  # None = huggingface default (CPU); "cuda" for GPU

ESTIMATORS = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]

# 30 short IMDB-style review excerpts (one local explanation each), matching the
# DistilBERT value function used for the paper's Table-1 language column.
SENTIMENT_TEXTS: tuple[str, ...] = (
    "This movie was absolutely wonderful and a real delight to watch",
    "A boring and predictable film that wasted two hours of my time",
    "The acting was superb and the story kept me hooked throughout",
    "Terrible script weak characters and an ending that made no sense",
    "An emotional and beautifully shot drama with a powerful message",
    "The plot dragged badly and the dialogue felt completely unnatural",
    "A charming comedy with sharp writing and a wonderful lead performance",
    "Dull lifeless and forgettable from the opening scene to the credits",
    "Stunning visuals paired with a haunting and unforgettable soundtrack",
    "The pacing was off and the film never found its emotional footing",
    "A thrilling adventure that delivers excitement in every single scene",
    "Cheap effects bad acting and a story that simply did not work",
    "Heartwarming and funny this film is a genuine crowd pleasing success",
    "Overlong messy and far too pleased with its own clever ideas",
    "A tense and gripping thriller that surprised me right until the end",
    "The characters were flat and I never cared what happened to them",
    "Brilliant direction and a career best performance from the lead actor",
    "A tedious slog with no humor no heart and no reason to exist",
    "Funny touching and smart this is easily the best film this year",
    "The story collapsed in the third act into a pile of cliches",
    "An inventive and bold film that takes real risks and earns them",
    "Painfully slow and self indulgent with nothing meaningful to say",
    "Gorgeous cinematography supports a quiet and deeply moving human story",
    "Loud obnoxious and utterly devoid of any genuine emotional weight",
    "A clever and witty screenplay elevates this small budget gem nicely",
    "The twist was obvious the acting wooden and the music intrusive",
    "Warm generous and wise this film treats its audience with respect",
    "A confused mess that mistakes noise and chaos for actual tension",
    "Beautifully acted and carefully written it lingers long after viewing",
    "Forgettable bland and stitched together from far better movies",
)


def make_estimator(name: str, n: int):
    """Construct an SV approximator in first-order (index="SV") mode."""
    if name == "MSR":
        return UnbiasedKernelSHAP(n=n, index="SV", max_order=1, random_state=0)
    if name == "SVARM":
        return SVARM(n=n, random_state=0)
    if name == "PermSamp":
        return PermutationSamplingSV(n=n, random_state=0)
    if name == "KernelSHAP":
        return KernelSHAP(n=n, random_state=0)
    if name == "kADDSHAP":
        return kADDSHAP(n=n, max_order=2, random_state=0)
    if name == "RegressionMSR":
        return RegressionMSR(n=n, index="SV", random_state=0)
    return OddSHAP(n=n, random_state=0)


def single_feature_values(interaction_values, n: int) -> np.ndarray:
    """Extract the first-order (singleton) Shapley values as a length-n vector."""
    return np.array([float(interaction_values.dict_values.get((i,), 0.0)) for i in range(n)])


def main() -> None:
    medians: dict[str, list[float]] = {est: [] for est in ESTIMATORS}
    for idx, text in enumerate(SENTIMENT_TEXTS[:N_INSTANCES]):
        game = SentimentAnalysis(input_text=text, device=DEVICE)
        n = game.n_players
        # Exact Shapley ground truth via the 2**n path (feasible for small d).
        exact = single_feature_values(ExactComputer(game, n_players=n)(index="SV"), n)
        budget = max(n + 1, 100 * n)
        for est in ESTIMATORS:
            iv = make_estimator(est, n).approximate(budget, game)
            mse = float(np.mean((single_feature_values(iv, n) - exact) ** 2))
            medians[est].append(mse)
        print(f"  [{idx + 1}/{N_INSTANCES}] n={n:2d}  OddSHAP MSE={medians['OddSHAP'][-1]:.3e}")

    print("\nDistilBERT value function — median MSE vs exact Shapley values "
          f"(N={N_INSTANCES}, budget=100*d)")
    median_mse = {est: float(np.median(medians[est])) for est in ESTIMATORS}
    ranks: dict[str, int] = {}
    order = sorted(ESTIMATORS, key=lambda e: median_mse[e])
    for rank, est in enumerate(order, start=1):
        ranks[est] = rank
    for est in order:
        print(f"  {est:16s} median MSE = {median_mse[est]:.3e}   rank {ranks[est]}")
    print(f"\nOddSHAP rank: {ranks['OddSHAP']}  (1 = best)")


if __name__ == "__main__":
    main()
