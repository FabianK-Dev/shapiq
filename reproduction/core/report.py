"""Shared plotting / reporting helpers for the reproduction notebooks.

Every figure carries its experimental context in the title and an info banner, so a
reader never has to guess the value-function family, ground-truth method, budget, or
which OddSHAP variant produced it. Data is read from the (gitignored) data directory the
cluster scripts write into.
"""

from __future__ import annotations

import csv
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .constants import (
    ESTIMATORS, ETA_BUDGETS, ETAS, GPU_VF_NAMES, PAPER_D, TABULAR_VF_NAMES,
    VARIANT_LABEL, VARIANT_SHORT,
)
from .style import (  # noqa: F401  (re-exported for the notebooks)
    OKABE_ITO, ODDSHAP_COLOR, PAPER_COLOR, PAPER_VF_ALIAS, estimator_style, paper_style,
    paper_vf_style, variant_style, vf_style,
)


def data_dir() -> Path:
    for c in (Path("reproduction/data"), Path("data"), Path("../data")):
        if c.is_dir():
            return c
    return Path("reproduction/data")


def paper_dir() -> Path:
    for c in (Path("reproduction/paper_reference"), Path("paper_reference"), Path("../paper_reference")):
        if c.is_dir():
            return c
    return Path("reproduction/paper_reference")


def read(name: str):
    with open(data_dir() / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has(name: str) -> bool:
    return (data_dir() / name).exists()


def load_paper_fig2(vf: str):
    """Paper's extracted Figure-2 curves for one value function: {method: [(budget, mse)]}.

    Resolves our value-function id to the paper's alias (e.g. realestate -> estate).
    """
    path = paper_dir() / "paper_fig2_extracted.csv"
    if not path.exists():
        return None
    pvf = PAPER_VF_ALIAS.get(vf, vf)
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["value_function"] == pvf:
                out.setdefault(r["method"], []).append((float(r["budget"]), float(r["mse"])))
    return {m: sorted(v) for m, v in out.items()} or None


def paper_figure_path(vf: str, kind: str = "fig2") -> Path | None:
    """Path to the paper's original figure PNG for a value function, if present.

    Our PNGs are saved under the full VF id (paper_fig2_realestate.png), but the paper's
    own short aliases (estate/cg60/il60) are also accepted — try both, full name first.
    """
    for name in (vf, PAPER_VF_ALIAS.get(vf, vf)):
        p = paper_dir() / "figures" / f"paper_{kind}_{name}.png"
        if p.exists():
            return p
    return None


def paper_legend_path() -> Path | None:
    """The paper's own Figure-2 legend strip (method name + colour key), if present.
    Shown above the panels so the paper's original figure (which has no per-panel legend)
    is decodable — its colours are the same as our paper-aligned reproduction panel."""
    p = paper_dir() / "figures" / "paper_fig2_legend.png"
    return p if p.exists() else None


def load_paper_oddshap_band(vf: str):
    """Digitised paper OddSHAP IQR band: list of (budget, median, q1, q3), or None.

    Read from paper_fig2_oddshap_band.csv (the band edges digitised from the paper's
    Figure 2 — approximate). Returns only rows with a valid q1 <= q3.
    """
    path = paper_dir() / "paper_fig2_oddshap_band.csv"
    if not path.exists():
        return None
    pvf = PAPER_VF_ALIAS.get(vf, vf)
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["value_function"] != pvf or not r.get("q1") or not r.get("q3"):
                continue
            b, m, q1, q3 = float(r["budget"]), float(r["median"]), float(r["q1"]), float(r["q3"])
            if q1 <= m <= q3:            # drop digitisation points where the band crossed
                out.append((b, m, q1, q3))
    return sorted(out) or None


def experiment_setup(variant: str, *, instances: int = 30) -> str:
    """A full, self-documenting description of the experimental protocol, so a reader knows
    exactly how every number was produced without opening the cluster scripts. Data-driven
    from the shared constants, so it can never drift from what actually ran."""
    try:
        import shapiq
        shapiq_v = getattr(shapiq, "__version__", "?")
    except Exception:  # noqa: BLE001
        shapiq_v = "?"
    tab = ", ".join(f"{v} (d={PAPER_D.get(v, '?')})" for v in TABULAR_VF_NAMES)
    gpu = ", ".join(f"{v} (d={PAPER_D.get(v, '?')})" for v in GPU_VF_NAMES)
    return "\n".join([
        "=" * 24 + "  EXPERIMENTAL SETUP  " + "=" * 24,
        f"  OddSHAP variant     : {VARIANT_LABEL.get(variant, variant)}",
        f"  Instances / VF      : N = {instances}   (every median / IQR band / mean / std is over these {instances})",
        f"  Tabular VFs ({len(TABULAR_VF_NAMES)})     : {tab}",
        f"  Deep-learning VFs ({len(GPU_VF_NAMES)}) : {gpu}",
        f"  Estimators ({len(ESTIMATORS)})       : {', '.join(ESTIMATORS)}",
        "  Value function      : XGBoost + interventional imputation, 50 background samples [tabular];",
        "                        ViT16 / DistilBERT class probability [deep-learning]",
        "  Ground truth        : exact interventional TreeSHAP [tabular];  exact 2^d enumeration [deep-learning]",
        "  Metric              : single-feature Shapley MSE vs ground truth, aggregated over the N instances",
        "  Table 1 / 3 budget  : m = max(d+1, 100*d)",
        "  Figure 2 budgets    : log-spaced grid, m = d+1 .. min(2^d, 20000)",
        f"  Figure 4 / 11       : fixed m in {ETA_BUDGETS} samples,  eta in {ETAS}  (# odd interactions = ceil(m/eta));",
        "                        y = MSE ratio vs the interaction-free baseline (LeverageSHAP)",
        "  Hardware            : LMU CIP cluster - CPU nodes (tabular) / NVIDIA GPU (deep-learning VFs)",
        f"  Software            : shapiq {shapiq_v}, python {platform.python_version()}, {platform.system()}",
        "=" * 70,
    ])


def environment_banner(variant: str, *, gt: str, vf_family: str = "XGBoost + interventional (50 bg)") -> str:
    """One-line experimental-environment string embedded under every figure group."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (f"variant: {VARIANT_LABEL.get(variant, variant)}  |  value function: {vf_family}  "
            f"|  ground truth: {gt}  |  N=30 instances  |  metric: Shapley MSE (median + IQR)  "
            f"|  python {platform.python_version()} / {platform.system()}  |  {stamp}")


def fig_title(base: str, vf: str, variant: str, extra: str = "") -> str:
    """Figure title that always names the value function (with d) and the variant."""
    d = PAPER_D.get(vf)
    dtag = f" (d={d})" if d else ""
    vtag = VARIANT_SHORT.get(variant, variant)
    return f"{base} — {vf}{dtag} · {vtag}{(' · ' + extra) if extra else ''}"


def add_banner(fig, text: str) -> None:
    """Attach the environment banner as a footnote below the figure.

    Reserves bottom margin so the 7pt caption is not clipped on export (a bare
    ``fig.text`` below the axes clips unless the saver uses bbox_inches='tight').
    """
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=7, color="#555555", wrap=True)


def load_table1(vf: str, variant: str):
    """Return {estimator: (median, q1, q3, mean, std)} for one vf/variant, or None."""
    name = f"table1_{vf}_{variant}.csv"
    if not has(name):
        return None
    out = {}
    for r in read(name):
        out[r["estimator"]] = (float(r["median"]), float(r["q1"]), float(r["q3"]),
                               float(r["mean"]), float(r["std"]))
    return out


# A Shapley single-feature MSE is O(1) at worst; values far above this are not a real
# estimate but a numerically diverged linear solve (e.g. kADDSHAP's normal-equations solve
# goes near-singular at low budget on high-dim VFs, giving MSE 1e31–1e94). We do NOT hide
# these — the notebook plots them (they shoot off-scale) and annotates them, because the
# divergence itself is a finding (a sibling of the TreeSHAP Vandermonde fix, PR #547).
DIVERGED_MSE = 1e6


def load_fig2(vf: str, variant: str, *, drop_diverged: bool = False):
    name = f"fig2_{vf}_{variant}.csv"
    if not has(name):
        return None
    out = {}
    for r in read(name):
        med = float(r["median"])
        if drop_diverged and (not np.isfinite(med) or med > DIVERGED_MSE):
            continue
        out.setdefault(r["estimator"], {})[int(r["budget"])] = (med, float(r["q1"]), float(r["q3"]))
    return out


def load_eta(vf: str, variant: str, budget: int = 10_000):
    """Figure-4 eta ablation for one vf/variant/budget.

    Returns a list of (n_interactions, ratio_median, ratio_q1, ratio_q3) sorted by
    n_interactions. ratio_q1/q3 fall back to the median when the band columns are absent
    (older data), so the caller can always unpack four values.
    """
    name = f"eta_{vf}_{variant}.csv"
    if not has(name):
        return None
    pts = []
    for r in read(name):
        if int(r["budget"]) == budget and r["eta"] != "base":
            ni = int(r["n_interactions"]); ratio = float(r["ratio_vs_base"])
            q1 = float(r["ratio_q1"]) if r.get("ratio_q1") else ratio
            q3 = float(r["ratio_q3"]) if r.get("ratio_q3") else ratio
            pts.append((ni, ratio, q1, q3))
    return sorted(pts)


def load_runtime(vf: str, variant: str):
    """Return {estimator: [(budget, median_runtime_s), ...]} for one vf/variant, or None."""
    name = f"runtime_{vf}_{variant}.csv"
    if not has(name):
        return None
    out = {}
    for r in read(name):
        out.setdefault(r["estimator"], []).append((int(r["budget"]), float(r["median_runtime_s"])))
    return {e: sorted(v) for e, v in out.items()}


def average_rank(vfs, variant: str):
    """Average rank of each estimator over the given value functions (median MSE)."""
    ranks = {e: [] for e in ESTIMATORS}
    used = []
    for vf in vfs:
        t = load_table1(vf, variant)
        if not t:
            continue
        used.append(vf)
        order = sorted((e for e in ESTIMATORS if e in t), key=lambda e: t[e][0])
        for rank, e in enumerate(order, 1):
            ranks[e].append(rank)
    return {e: float(np.mean(v)) for e, v in ranks.items() if v}, used


def table3_dataframe(vfs, variant: str, statistics=("median", "mean", "std", "q1", "q3")):
    """Table 3 — summary statistics of Shapley MSE, rows = (estimator, statistic), cols = VF.

    Reproduces the paper's Table-3 layout (per-estimator mean/median/q1/q3) and additionally
    reports the **standard deviation** (``std``), which the paper's own table omits. Falls
    back to a nested dict of strings if pandas is unavailable.
    """
    # (estimator, statistic) -> {vf_column: value}
    data: dict = {}
    est_seen: list = []
    cols: list = []
    for vf in vfs:
        t = load_table1(vf, variant)
        if not t:
            continue
        col = f"{vf} (d={PAPER_D.get(vf, '?')})"
        cols.append(col)
        for e, (m, q1, q3, mean, std) in t.items():
            if e not in est_seen:
                est_seen.append(e)
            vals = {"median": m, "mean": mean, "std": std, "q1": q1, "q3": q3}
            for s in statistics:
                data.setdefault((e, s), {})[col] = vals[s]
    order = sorted(est_seen, key=lambda e: (0 if e == "OddSHAP" else 1, e))
    try:
        import pandas as pd

        rows = [(e, s) for e in order for s in statistics]
        df = pd.DataFrame(index=pd.MultiIndex.from_tuples(rows, names=["estimator", "statistic"]),
                          columns=cols, dtype=object)
        for (e, s) in rows:
            for col, v in data.get((e, s), {}).items():
                df.loc[(e, s), col] = f"{v:.2e}"
        return df.fillna("")
    except ImportError:
        return {f"{e}/{s}": data.get((e, s), {}) for e in order for s in statistics}
