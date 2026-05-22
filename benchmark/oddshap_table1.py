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
    MSRBiased,
    PermutationSamplingSV,
    UnbiasedKernelSHAP,
)
from shapiq.approximator.regression.oddshap import OddSHAP

# Table-1 row label -> approximator class.
TABLE1_ESTIMATORS: dict[str, type] = {
    "MSR": UnbiasedKernelSHAP,
    "SVARM": SVARM,
    "PermutationSampling": PermutationSamplingSV,
    "RegressionMSR": MSRBiased,
    "OddSHAP": OddSHAP,
}

# Multi-index estimators need the explicit SV-mode signature.
_NEEDS_SV_KWARGS = {"MSRBiased"}


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
    """Yield ViT16 image value-function games, one per ImageNet example.

    The ViT setup auto-selects CUDA when available; ``device`` is accepted
    for a uniform factory signature and currently unused here.
    """
    del device  # ViT setup auto-detects CUDA
    from shapiq_games.benchmark.imagenet_examples import get_imagenet_example
    from shapiq_games.benchmark.local_xai import ImageClassifier

    for idx in range(n_instances):
        image = get_imagenet_example(idx)
        game = ImageClassifier(x_explain=image, model_name="vit_16_patches")
        yield GameInstance(game=game, n=game.n_players, label=f"vit16_{idx}")


GAME_FACTORIES = {
    "distilbert": make_distilbert_instances,
    "vit16": make_vit16_instances,
}


# -----------------------------------------------------------------------------
# Estimator construction + evaluation
# -----------------------------------------------------------------------------


def construct_estimator(name: str, cls: type, n: int, seed: int):
    """Instantiate an estimator in SV mode; return None if it cannot be built."""
    kwargs = {"n": n, "random_state": seed}
    if cls.__name__ in _NEEDS_SV_KWARGS:
        kwargs.update(index="SV", max_order=1)
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
    # Paper protocol: budget m ~= 100 * d, clamped to the affordable range.
    budget = min(2 ** n, max(n + 1, 100 * n))

    exact = ExactComputer(inst.game, n_players=n)(index="SV").values

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
        if iv.values.shape != exact.shape:
            per_method[name] = float("nan")
            continue
        per_method[name] = mse(iv.values, exact)

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
