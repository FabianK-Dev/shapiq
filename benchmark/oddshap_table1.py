"""OddSHAP paper Table 1 reproduction harness.

Reproduces the **OddSHAP row** of Table 1 from arXiv:2602.01399 ("An Odd
Estimator for Shapley Values") — the average MSE of estimated versus exact
Shapley values at budget ``m ~= 100 * d``, aggregated over several local
explanations.

Scope
-----
Table 1 in the paper is an 8-value-function x 7-estimator grid. This script
is deliberately scoped to the **OddSHAP-related wiring**: it runs OddSHAP
and the SV baselines that already ship with shapiq (MSR / SVARM /
PermutationSampling / RegressionMSR) against exact ground truth, in the
paper's protocol, and prints the result in Table-1 / Table-3 format.

LeverageSHAP and the 3-PolySHAP variants are *not* run here — they live on
separate feature branches and are not yet registered in
``SV_APPROXIMATORS``. Once all three new approximators land on one branch,
they are picked up automatically (the estimator list is read dynamically).

Ground truth
------------
Exact Shapley values come from ``shapiq.ExactComputer`` evaluated on the
*game itself*. This is the only unambiguously exact ground truth: it
matches whatever value function (imputer, normalization) the estimators
see. It is feasible only for small ``d`` — hence this harness targets the
paper's small-``d`` language / image value functions (DistilBERT ``d~=14``,
ViT16 ``d=16``), where ``2**d`` game evaluations are affordable. For the
tabular value functions (``d`` 30-101) exact ground truth needs the
interventional-TreeSHAP path, which is left for the full reproduction.

Estimator -> Table 1 row mapping
--------------------------------
    MSR                  -> UnbiasedKernelSHAP
    SVARM                -> SVARM
    PermutationSampling  -> PermutationSamplingSV
    RegressionMSR        -> MSRBiased
    OddSHAP              -> OddSHAP                (the row of interest)

Run::

    python -m benchmark.oddshap_table1 --game distilbert --instances 5
    python -m benchmark.oddshap_table1 --game distilbert --instances 30 \\
        --output benchmark/results/oddshap_table1.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shapiq import ExactComputer

# Estimators that map onto Table 1 rows and are registered in shapiq today.
# OddSHAP is imported from its own module so the harness works on the
# oddshap_approximator branch even before OddSHAP is added to the public
# package __init__.
from shapiq.approximator import (
    SVARM,
    PermutationSamplingSV,
    UnbiasedKernelSHAP,
)
from shapiq.approximator.regression.oddshap import OddSHAP

# Table-1 row label -> approximator class.
#
# OddSHAP is the row under verification. MSR / SVARM / PermutationSampling are
# shapiq's own, well-established implementations and act as the harness sanity
# check: if they reproduce the paper, the harness and protocol are sound, so
# the OddSHAP row can be read as a verdict on our OddSHAP implementation.
#
# Table 1's "RegressionMSR" row is intentionally NOT run here: the paper's
# RegressionMSR is the authors' own `shapiq.approximator.regressionMSR`
# class (vendored in their PolySHAP repo), which is a different estimator
# from upstream shapiq's `MSRBiased` — mapping it to `MSRBiased` gave a
# stable ~1300x mismatch. It is a baseline, not OddSHAP, so it is dropped
# rather than mis-mapped.
TABLE1_ESTIMATORS: dict[str, type] = {
    "MSR": UnbiasedKernelSHAP,
    "SVARM": SVARM,
    "PermutationSampling": PermutationSamplingSV,
    "OddSHAP": OddSHAP,
}

# Multi-index estimators that need the explicit SV-mode signature (none today).
_NEEDS_SV_KWARGS: set[str] = set()


# -----------------------------------------------------------------------------
# Input instances for the language value function
# -----------------------------------------------------------------------------

# 30 short IMDB-style review excerpts used as local-explanation instances for
# the DistilBERT sentiment value function. Token counts vary by a few tokens;
# the budget m = 100 * d is computed per instance from the realised d.
_SENTIMENT_TEXTS: tuple[str, ...] = (
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


# -----------------------------------------------------------------------------
# Game factory
# -----------------------------------------------------------------------------


@dataclass
class GameInstance:
    game: object
    n: int
    label: str


def make_distilbert_instances(n_instances: int, device: str | None = None):
    """Yield DistilBERT sentiment value-function games, one per review text.

    ``device`` is forwarded to the shapiq SentimentAnalysis game — pass
    ``"cuda"`` (or a GPU index) to run the DistilBERT forward passes on a
    GPU, which is the dominant cost of the exact ground-truth computation.
    """
    from shapiq_games.benchmark.local_xai import SentimentAnalysis

    texts = _SENTIMENT_TEXTS[:n_instances]
    for idx, text in enumerate(texts):
        game = SentimentAnalysis(input_text=text, device=device)
        yield GameInstance(game=game, n=game.n_players, label=f"distilbert_{idx}")


def make_vit16_instances(n_instances: int, device: str | None = None):
    """Yield ViT16 image value-function games, one per bundled ImageNet example.

    Uses the ``vit_16_patches`` model (4x4 grid -> d=16 players), matching the
    paper's ``ViT4by4Patches`` value function. The ViT setup auto-selects CUDA
    when available; ``device`` is accepted for a uniform factory signature.
    """
    del device  # ViT setup auto-detects CUDA
    from pathlib import Path as _Path

    import shapiq_games
    from shapiq_games.benchmark.local_xai import ImageClassifier

    img_dir = _Path(shapiq_games.__file__).parent / "benchmark" / "imagenet_examples"
    images = sorted(img_dir.glob("*.JPEG")) + sorted(img_dir.glob("*.jpg"))
    for idx, img_path in enumerate(images[:n_instances]):
        game = ImageClassifier(
            x_explain_path=str(img_path), model_name="vit_16_patches",
        )
        yield GameInstance(game=game, n=game.n_players, label=f"vit16_{idx}")


def _make_tabular_factory(name: str):
    """Adapt a paper tabular value function to the (n_instances, device) API.

    The tabular value functions are path-dependent TreeSHAP games over a
    RandomForest (see ``_paper_datasets``); ground truth is the game's own
    polynomial TreeSHAP-IQ ``exact_values``, so d up to 101 is feasible.
    """
    def factory(n_instances: int, device: str | None = None):
        del device  # tree games are CPU-only
        from _paper_datasets import make_tabular_games

        for game, label in make_tabular_games(name, n_instances):
            yield GameInstance(game=game, n=game.n_players, label=label)

    return factory


GAME_FACTORIES = {
    "distilbert": make_distilbert_instances,
    "vit16": make_vit16_instances,
}
# Tabular paper value functions (Estate / Cancer / IL60 / CG60 / NHANES /
# Crime) — registered dynamically so the choice list stays in one place.
for _tabular_name in ("realestate", "cancer", "independentlinear60",
                      "corrgroups60", "nhanes", "crime"):
    GAME_FACTORIES[_tabular_name] = _make_tabular_factory(_tabular_name)


# -----------------------------------------------------------------------------
# Estimator construction + evaluation
# -----------------------------------------------------------------------------


# OddSHAP proxy interaction-screening order. The class defaults
# proxy_max_order to n, which makes InterventionalTreeExplainer build a full
# 2**n interaction lookup inside _select_odd_interactions -> MemoryError for
# d >= ~25. Order-3 is the dominant higher-order odd interaction and keeps the
# lookup tractable up to d=101 (C(101,<=3) ~ 1.7e5). Capped here so the same
# OddSHAP configuration runs across every Table-1 value function.
ODDSHAP_PROXY_MAX_ORDER = 3


def construct_estimator(name: str, cls: type, n: int, seed: int):
    """Instantiate an estimator in SV mode; return None if it cannot be built."""
    kwargs = {"n": n, "random_state": seed}
    if cls.__name__ in _NEEDS_SV_KWARGS:
        kwargs.update(index="SV", max_order=1)
    if cls is OddSHAP:
        kwargs["proxy_max_order"] = min(ODDSHAP_PROXY_MAX_ORDER, n)
    try:
        return cls(**kwargs)
    except TypeError:
        try:
            return cls(n=n, random_state=seed)
        except (TypeError, ValueError):
            return None
    except ValueError:
        return None


def mse(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.mean((estimated - ground_truth) ** 2))


def _singletons(iv, n: int) -> np.ndarray:
    """Extract the n order-1 (singleton) Shapley values as a flat vector.

    Normalises both ground truth and estimator output to the same length-n
    representation regardless of whether the InteractionValues object also
    carries an order-0 baseline term.
    """
    return np.array([float(iv[(i,)]) for i in range(n)], dtype=float)


def exact_ground_truth(game, n: int) -> np.ndarray:
    """Exact singleton Shapley values for the game.

    ``TreeSHAPIQXAI`` games carry a polynomial TreeSHAP-IQ ``exact_values``
    that is feasible for any d (up to the paper's d=101); every other game
    uses the ``ExactComputer`` 2**d path, feasible only for small d such as
    DistilBERT (d=14) and ViT16 (d=16).
    """
    from shapiq_games.benchmark.treeshapiq_xai import TreeSHAPIQXAI

    if isinstance(game, TreeSHAPIQXAI):
        iv = game.exact_values(index="SV", order=1)
    else:
        iv = ExactComputer(game, n_players=n)(index="SV")
    return _singletons(iv, n)


# -----------------------------------------------------------------------------
# Single-instance evaluation
# -----------------------------------------------------------------------------


@dataclass
class InstanceResult:
    label: str
    n: int
    budget: int
    per_method_mse: dict[str, float]  # method -> MSE (NaN if skipped)


def evaluate_instance(inst: GameInstance, seed: int = 0) -> InstanceResult:
    """Compute exact ground truth and every estimator's MSE for one game."""
    n = inst.n
    # Paper protocol: budget m ~= 100 * d, clamped to the affordable range
    # (>= n+1 so every estimator can run, <= 2**d, <= 20000 as in the paper).
    budget = min(2 ** n, 20000, max(n + 1, 100 * n))

    exact = exact_ground_truth(inst.game, n)

    per_method: dict[str, float] = {}
    for name, cls in TABLE1_ESTIMATORS.items():
        estimator = construct_estimator(name, cls, n, seed)
        if estimator is None:
            per_method[name] = float("nan")
            continue
        try:
            iv = estimator.approximate(budget, inst.game)
        except (ValueError, RuntimeError):
            per_method[name] = float("nan")
            continue
        try:
            approx = _singletons(iv, n)
        except (KeyError, IndexError, TypeError):
            per_method[name] = float("nan")
            continue
        per_method[name] = mse(approx, exact)

    return InstanceResult(
        label=inst.label, n=n, budget=budget, per_method_mse=per_method,
    )


# -----------------------------------------------------------------------------
# Aggregation -> Table 1 / Table 3 format
# -----------------------------------------------------------------------------


def summarise(results: list[InstanceResult]) -> dict[str, dict[str, float]]:
    """Per-method mean / 1st-quartile / median / 3rd-quartile MSE."""
    summary: dict[str, dict[str, float]] = {}
    for name in TABLE1_ESTIMATORS:
        values = [
            r.per_method_mse[name]
            for r in results
            if not np.isnan(r.per_method_mse.get(name, float("nan")))
        ]
        if not values:
            summary[name] = {
                "mean": float("nan"), "q1": float("nan"),
                "median": float("nan"), "q3": float("nan"), "n_ok": 0,
            }
            continue
        summary[name] = {
            "mean": float(np.mean(values)),
            "q1": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "q3": float(np.percentile(values, 75)),
            "n_ok": len(values),
        }
    return summary


def print_table(summary: dict[str, dict[str, float]], game_name: str) -> None:
    print()
    print(f"OddSHAP Table 1 reproduction — {game_name}")
    print(f"{'Estimator':<22} {'Mean MSE':>12} {'Q1':>12} "
          f"{'Median':>12} {'Q3':>12} {'n_ok':>6}")
    print("-" * 80)
    for name, stats in summary.items():
        print(f"{name:<22} {stats['mean']:>12.3e} {stats['q1']:>12.3e} "
              f"{stats['median']:>12.3e} {stats['q3']:>12.3e} "
              f"{stats['n_ok']:>6}")


def write_csv(results: list[InstanceResult], path: Path) -> None:
    """Long-format CSV: one row per (instance, method)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "n", "budget", "method", "mse"])
        for r in results:
            for name, value in r.per_method_mse.items():
                writer.writerow([r.label, r.n, r.budget, name, value])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OddSHAP paper Table 1 reproduction (OddSHAP-scoped).",
    )
    parser.add_argument(
        "--game", default="distilbert", choices=sorted(GAME_FACTORIES),
        help="Value function to reproduce (default: distilbert).",
    )
    parser.add_argument(
        "--instances", default=5, type=int,
        help="Number of local-explanation instances (paper uses 30).",
    )
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Random seed for the estimators.",
    )
    parser.add_argument(
        "--output", default="benchmark/results/oddshap_table1.csv", type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for the value-function model, e.g. 'cuda' or 'cpu'. "
             "Defaults to the library default (CPU).",
    )
    args = parser.parse_args(argv)

    factory = GAME_FACTORIES[args.game]
    results: list[InstanceResult] = []
    print(f"Reproducing OddSHAP Table 1 row on '{args.game}', "
          f"{args.instances} instances, device={args.device or 'default'} ...",
          file=sys.stderr)

    for i, inst in enumerate(factory(args.instances, args.device), start=1):
        t0 = time.perf_counter()
        result = evaluate_instance(inst, seed=args.seed)
        results.append(result)
        odd = result.per_method_mse.get("OddSHAP", float("nan"))
        print(f"  [{i}/{args.instances}] {inst.label:<16} n={inst.n:>3} "
              f"budget={result.budget:>6}  OddSHAP MSE={odd:.3e}  "
              f"({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    if not results:
        print("No instances evaluated.", file=sys.stderr)
        return 1

    write_csv(results, args.output)
    summary = summarise(results)
    print_table(summary, args.game)
    print(f"\nCSV written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
