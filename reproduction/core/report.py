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

from .constants import ESTIMATORS, PAPER_D, VARIANT_LABEL, VARIANT_SHORT
from .style import (  # noqa: F401  (re-exported for the notebooks)
    OKABE_ITO, ODDSHAP_COLOR, estimator_style, variant_style, vf_style,
)


def data_dir() -> Path:
    for c in (Path("reproduction/data"), Path("data"), Path("../data")):
        if c.is_dir():
            return c
    return Path("reproduction/data")


def read(name: str):
    with open(data_dir() / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has(name: str) -> bool:
    return (data_dir() / name).exists()


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


def load_fig2(vf: str, variant: str):
    name = f"fig2_{vf}_{variant}.csv"
    if not has(name):
        return None
    out = {}
    for r in read(name):
        out.setdefault(r["estimator"], {})[int(r["budget"])] = (
            float(r["median"]), float(r["q1"]), float(r["q3"]))
    return out


def load_eta(vf: str, variant: str, budget: int = 10_000):
    name = f"eta_{vf}_{variant}.csv"
    if not has(name):
        return None
    pts = []
    for r in read(name):
        if int(r["budget"]) == budget and r["eta"] != "base":
            pts.append((int(r["n_interactions"]), float(r["ratio_vs_base"])))
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


def table3_dataframe(vfs, variant: str):
    """Table 3 as a styled pandas DataFrame: rows = estimators, cols = 'vf median [Q1,Q3]'.

    Falls back to a plain dict of strings if pandas is unavailable.
    """
    cells = {}
    for vf in vfs:
        t = load_table1(vf, variant)
        if not t:
            continue
        for e, (m, q1, q3, _mean, _std) in t.items():
            cells.setdefault(e, {})[f"{vf} (d={PAPER_D.get(vf, '?')})"] = f"{m:.2e} [{q1:.1e}, {q3:.1e}]"
    order = sorted(cells, key=lambda e: (0 if e == "OddSHAP" else 1, e))
    try:
        import pandas as pd

        df = pd.DataFrame({e: cells[e] for e in order}).T
        return df
    except ImportError:
        return {e: cells[e] for e in order}
