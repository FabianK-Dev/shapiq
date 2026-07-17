"""Experiment: OddSHAP PR #522 (ours, merged) vs PR #560 (author improvement).

Runs both vendored OddSHAP revisions through the *identical* value function, ground
truth, and budget grid, and reports where their numbers differ. The finding: they are
bitwise-identical at every budget >= n*interaction_factor; PR #560's entire contribution
is the low-budget regime (budget < n*eta), where PR #522 refuses and #560 still estimates.

Run:  uv run python notebooks/oddshap/experiment_variant_delta.py   [--vf realestate] [--instances 5]
"""

from __future__ import annotations

import argparse

import numpy as np

import shapiq_games.datasets as datasets

from .core.harness import load_oddshap, log_budgets, prepare_vf, sfv, warm_dispatch

_LOADERS = {
    "realestate": (datasets.load_real_estate, "continuous"),
    "cancer": (datasets.load_breast_cancer, "native_binary"),
    "nhanes": (datasets.load_nhanesi, "continuous"),
}


def variant_delta(vf_name: str, n_instances: int):
    loader, kind = _LOADERS[vf_name]
    vf = prepare_vf(loader, kind, classifier=(kind != "native_binary"))
    budgets = sorted(set(log_budgets(vf.n)) | {20, 50, 100, vf.n * 10})
    rows = []
    for b in budgets:
        v522, v560 = [], []
        for i in range(min(n_instances, vf.x_test.shape[0])):
            target = vf.x_test[i]
            truth = vf.ground_truth(target)
            game = vf.game(target)
            for tag, sink in (("v522_merged", v522), ("v560_improved", v560)):
                try:
                    iv = load_oddshap(tag)(n=vf.n, random_state=0).approximate(b, game)
                    sink.append(float(np.mean((sfv(iv, vf.n) - truth) ** 2)))
                except (ValueError, RuntimeError):
                    sink.append(float("nan"))
        rows.append(
            (
                b,
                float(np.nanmedian(v522)) if v522 else float("nan"),
                float(np.nanmedian(v560)) if v560 else float("nan"),
            )
        )
    return vf, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", default="realestate", choices=list(_LOADERS))
    ap.add_argument("--instances", type=int, default=5)
    args = ap.parse_args()
    warm_dispatch()
    vf, rows = variant_delta(args.vf, args.instances)
    print(f"{args.vf} d={vf.n}: v522 min budget = n*eta = {vf.n * 10}; v560 min budget = eta = 10")
    print(f"{'budget':>8} {'v522 (ours)':>14} {'v560 (author)':>14}  {'note':>18}")
    for b, a, c in rows:
        note = (
            "identical"
            if np.isfinite(a) and np.isfinite(c) and abs(a - c) <= 1e-9 * max(a, c, 1e-30)
            else ("#560-only regime" if not np.isfinite(a) and np.isfinite(c) else "")
        )
        astr = "REFUSED" if not np.isfinite(a) else f"{a:.3e}"
        print(f"{b:>8} {astr:>14} {c:>14.3e}  {note:>18}")


if __name__ == "__main__":
    main()
