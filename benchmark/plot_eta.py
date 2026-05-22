"""Plot the OddSHAP interaction-count ablation — paper Figure 4 / 11.

Reads one or more ``oddshap_eta_<game>.csv`` files produced by
``benchmark/eta_ablation.py`` and renders a single figure: MSE ratio
(against each game's fewest-interactions setting) versus the mean number
of odd interactions, one curve per value function.

The paper's Figure-4 conclusion is the U-shape — MSE falls as odd
interactions are added, reaches a minimum, then rises as too many
interactions overfit.

Run::

    python benchmark/plot_eta.py --csv oddshap_eta_cancer.csv \\
        oddshap_eta_realestate.csv --out figure4_eta_ablation.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_eta(csv_path: Path):
    """Return sorted [(mean_interactions, mean_mse), ...] for one game CSV."""
    mse_by_factor: dict[int, list[float]] = defaultdict(list)
    cnt_by_factor: dict[int, list[float]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                factor = int(row["interaction_factor"])
                mse = float(row["mse"])
                count = float(row.get("n_active_interactions", "nan"))
            except (ValueError, TypeError, KeyError):
                continue
            if not math.isnan(mse):
                mse_by_factor[factor].append(mse)
            if not math.isnan(count):
                cnt_by_factor[factor].append(count)
    points = []
    for factor in sorted(mse_by_factor, reverse=True):  # few -> many
        mses = mse_by_factor[factor]
        cnts = cnt_by_factor.get(factor, [])
        if not mses or not cnts:
            continue
        points.append((sum(cnts) / len(cnts), sum(mses) / len(mses)))
    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot the OddSHAP interaction-count ablation (Figure 4).",
    )
    parser.add_argument("--csv", nargs="+", required=True, type=Path,
                        help="oddshap_eta_<game>.csv files.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output PNG path.")
    args = parser.parse_args(argv)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    plotted = 0
    for csv_path in args.csv:
        if not csv_path.exists():
            print(f"skip (not found): {csv_path}")
            continue
        points = load_eta(csv_path)
        if not points:
            continue
        game = csv_path.stem.replace("oddshap_eta_", "")
        counts = [p[0] for p in points]
        base_mse = points[0][1]  # fewest-interactions reference
        ratios = [p[1] / base_mse for p in points]
        ax.plot(counts, ratios, marker="o", markersize=5, label=game)
        plotted += 1

    if not plotted:
        print("No usable eta CSVs.")
        return 1

    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean number of odd interactions")
    ax.set_ylabel("MSE ratio vs fewest-interactions setting")
    ax.set_title("Figure 4 — OddSHAP interaction-count ablation")
    ax.legend(fontsize=9)
    ax.grid(visible=True, which="both", alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Figure 4 written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
