"""Plot OddSHAP paper Figures 2 / 3 / 5 from a budget-grid reproduction CSV.

Consumes a CSV produced by ``benchmark/oddshap_table1.py --budgets grid``
(columns ``instance,n,budget,method,mse,runtime``) and renders, for one
value function:

  Figure 2  MSE vs budget          (log-log)
  Figure 5  total runtime vs budget (log-log)
  Figure 3  MSE vs runtime          (log-log)

Each curve is a method; points are means over the local-explanation
instances. The paper's conclusions are that OddSHAP's MSE-vs-budget curve
sits below the baselines (Figure 2) and that its runtime stays competitive
(Figures 3 / 5).

Run::

    python benchmark/plot_figures.py --csv oddshap_table1_cancer_grid.csv \\
        --game cancer --out-dir documents/oddshap_paper/figures
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

METHOD_STYLE = {
    "OddSHAP": {"color": "#d62728", "marker": "o", "zorder": 5, "lw": 2.2},
    "MSR": {"color": "#1f77b4", "marker": "s", "zorder": 3, "lw": 1.5},
    "SVARM": {"color": "#2ca02c", "marker": "^", "zorder": 3, "lw": 1.5},
    "PermutationSampling": {"color": "#ff7f0e", "marker": "D",
                            "zorder": 3, "lw": 1.5},
}


# MSE below this is exact computation (budget >= 2**d), not approximation
# error — excluded so it does not distort an approximation-quality plot.
# This matters for value functions whose instances have variable d (e.g.
# DistilBERT, where token count differs per text): some instances reach
# exact computation while others do not.
_EXACT_MSE_FLOOR = 1e-12


def load_grid(csv_path: Path, only_n: int | None = None):
    """Return {method: {budget: (mean_mse, mean_runtime)}}.

    ``only_n`` keeps only instances of that dimension. This is needed for
    value functions whose instances have variable d (DistilBERT — token
    count differs per text): instances of different d have different
    budget grids, so averaging by exact budget mixes inconsistent instance
    subsets and the curve becomes jagged. Restricting to one d makes every
    kept instance share the same budget grid -> a clean averaged curve.
    """
    mse_acc: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    rt_acc: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            budget = int(row["budget"])
            if only_n is not None and int(row["n"]) != only_n:
                continue
            try:
                mse = float(row["mse"])
                runtime = float(row.get("runtime", "nan"))
            except (ValueError, TypeError):
                continue
            if not math.isnan(mse) and mse >= _EXACT_MSE_FLOOR:
                mse_acc[method][budget].append(mse)
            if not math.isnan(runtime):
                rt_acc[method][budget].append(runtime)

    out: dict[str, dict[int, tuple[float, float]]] = {}
    for method in mse_acc:
        out[method] = {}
        for budget in sorted(mse_acc[method]):
            mses = mse_acc[method][budget]
            rts = rt_acc[method].get(budget, [])
            mean_mse = sum(mses) / len(mses)
            mean_rt = (sum(rts) / len(rts)) if rts else float("nan")
            out[method][budget] = (mean_mse, mean_rt)
    return out


def _methods_in_order(grid: dict) -> list[str]:
    order = ["MSR", "SVARM", "PermutationSampling", "OddSHAP"]
    return [m for m in order if m in grid] + [
        m for m in grid if m not in order]


def plot_all(csv_path: Path, game: str, out_dir: Path,
             only_n: int | None = None) -> int:
    grid = load_grid(csv_path, only_n=only_n)
    if not grid:
        print(f"No usable rows in {csv_path}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = _methods_in_order(grid)

    def style(method: str) -> dict:
        return METHOD_STYLE.get(
            method, {"color": "gray", "marker": "x", "zorder": 2, "lw": 1.2})

    # ---- Figure 2: MSE vs budget -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for method in methods:
        pts = sorted(grid[method].items())
        budgets = [b for b, _ in pts]
        mses = [v[0] for _, v in pts]
        ax.plot(budgets, mses, label=method, markersize=5, **style(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("budget (model evaluations)")
    ax.set_ylabel("mean MSE vs exact Shapley values")
    ax.set_title(f"Figure 2 — MSE vs budget ({game})")
    ax.legend(fontsize=8)
    ax.grid(visible=True, which="both", alpha=0.25)
    fig.tight_layout()
    fig2 = out_dir / f"figure2_mse_vs_budget_{game}.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)

    # ---- Figure 5: total runtime vs budget ---------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    has_rt = False
    for method in methods:
        pts = sorted(grid[method].items())
        budgets = [b for b, v in pts if not math.isnan(v[1])]
        runtimes = [v[1] for _, v in pts if not math.isnan(v[1])]
        if not runtimes:
            continue
        has_rt = True
        ax.plot(budgets, runtimes, label=method, markersize=5, **style(method))
    if has_rt:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("budget (model evaluations)")
        ax.set_ylabel("mean runtime per explanation (s)")
        ax.set_title(f"Figure 5 — runtime vs budget ({game})")
        ax.legend(fontsize=8)
        ax.grid(visible=True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"figure5_runtime_vs_budget_{game}.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: MSE vs runtime ------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for method in methods:
        pts = sorted(grid[method].items())
        runtimes = [v[1] for _, v in pts if not math.isnan(v[1])]
        mses = [v[0] for _, v in pts if not math.isnan(v[1])]
        if not runtimes:
            continue
        ax.plot(runtimes, mses, label=method, markersize=5, **style(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean runtime per explanation (s)")
    ax.set_ylabel("mean MSE vs exact Shapley values")
    ax.set_title(f"Figure 3 — MSE vs runtime ({game})")
    ax.legend(fontsize=8)
    ax.grid(visible=True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"figure3_mse_vs_runtime_{game}.png", dpi=150)
    plt.close(fig)

    print(f"Figures written to {out_dir} (game={game})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot OddSHAP Figures 2/3/5 from a budget-grid CSV.",
    )
    parser.add_argument("--csv", required=True, type=Path,
                        help="Budget-grid reproduction CSV.")
    parser.add_argument("--game", required=True,
                        help="Value-function name, used in titles/filenames.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Directory for the PNG figures.")
    parser.add_argument("--only-n", type=int, default=None,
                        help="Keep only instances of this dimension d. Use "
                             "for variable-d value functions (DistilBERT) so "
                             "the averaged curve is not jagged.")
    args = parser.parse_args(argv)
    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        return 1
    return plot_all(args.csv, args.game, args.out_dir, only_n=args.only_n)


if __name__ == "__main__":
    raise SystemExit(main())
