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

# canonical per-method colours (shared across every notebook + the paper panels)
PCOL = {
    "MSR": (0.816, 0.816, 0.816), "SVARM": (0.549, 0.549, 0.549),
    "PermutationSampling": (0.333, 0.333, 0.333), "PermSamp": (0.333, 0.333, 0.333),
    "ProxyLGBM": (0.69, 0.745, 0.773), "RegressionMSR": (0.247, 0.318, 0.71),
    "3-PolySHAP": (0.486, 0.702, 0.259), "LeverageSHAP": (0.0, 0.588, 0.533),
    "KernelSHAP": (0.80, 0.60, 0.85), "kADDSHAP": (0.95, 0.70, 0.30),
    "OddSHAP": (0.902, 0.29, 0.098),
}
ESTIMATORS = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]

# paper d per value function (for titles)
PAPER_D = {"cancer": 30, "realestate": 15, "corrgroups60": 60, "independentlinear60": 60,
           "nhanes": 79, "crime": 101, "vit16": 16, "distilbert": 14}
VARIANT_LABEL = {"v522_merged": "OddSHAP PR #522 (ours, merged)",
                 "v560_improved": "OddSHAP PR #560 (author improvement)",
                 "library": "installed shapiq.OddSHAP"}


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
    vtag = {"v522_merged": "#522", "v560_improved": "#560", "library": "lib"}.get(variant, variant)
    return f"{base} — {vf}{dtag} · {vtag}{(' · ' + extra) if extra else ''}"


def add_banner(fig, text: str) -> None:
    """Attach the environment banner as a footnote below a figure."""
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=6.5, color="#555555", wrap=True)


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
