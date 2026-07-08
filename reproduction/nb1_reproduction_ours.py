# %% [markdown]
# # NB1 — OddSHAP paper reproduction with **our merged contribution (PR #522)**
#
# This notebook reproduces the paper's results (Fumagalli et al. 2026, *An Odd Estimator
# for Shapley Values*, arXiv:2602.01399) using the OddSHAP implementation **our group
# contributed and merged (PR #522)**. Every figure names its value function, dimension,
# and the OddSHAP variant, and carries an experimental-environment banner.
#
# > **Note on NB1 vs NB2.** NB1 and NB2 share the *same* code, parameterised by one
# > variant switch (`ODDSHAP_VARIANT`): NB1 = `v522_merged` (ours), NB2 = `v560_improved`
# > (the author's PR #560). They are structurally identical by design; NB3 shows the two
# > produce the same paper-scale numbers.
#
# **Provenance.** All numbers are produced by the cluster scripts in `reproduction/cluster/`
# and written to the gitignored `reproduction/data/` folder; this notebook reads them and
# plots. Regenerate with `bash reproduction/cluster/submit_all.sh v522_merged`.
#
# **Value function** (paper Table 2): XGBoost + interventional perturbation, 50 background
# samples; ground truth = exact interventional TreeSHAP (tabular) / exact enumeration (GPU).
#
# **Accessibility.** Colours are Okabe-Ito (colour-blind-safe) and every series also carries
# a distinct marker and line style, so figures read in greyscale and under any colour vision.

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path.cwd() if Path("reproduction").is_dir() else Path.cwd().parent))
from reproduction.core import report as R
from reproduction.core.constants import ESTIMATORS, GPU_VF_NAMES, TABULAR_VF_NAMES

VARIANT = os.environ.get("ODDSHAP_VARIANT", "v522_merged")
TAB_VFS, GPU_VFS = TABULAR_VF_NAMES, GPU_VF_NAMES
BANNER_TAB = R.environment_banner(VARIANT, gt="exact interventional TreeSHAP")
BANNER_GPU = R.environment_banner(VARIANT, gt="exact Shapley (2^d enumeration)",
                                  vf_family="deep model (ViT16 / DistilBERT), baseline imputation")
print(R.experiment_setup(VARIANT))
_missing = [vf for vf in TAB_VFS + GPU_VFS if not R.has(f"table1_{vf}_{VARIANT}.csv")]
print("  Data completeness   :",
      "all value functions present" if not _missing
      else f"MISSING {_missing} — run the cluster scripts to fill")

# %% [markdown]
# ## Table 1 — average rank (headline result)
#
# Median MSE at budget ≈ 100·d; each estimator's average rank across value functions is the
# paper's headline. OddSHAP (highlighted) reproduces rank-1.

# %%
avg_rank, used = R.average_rank(TAB_VFS + GPU_VFS, VARIANT)
if avg_rank:
    ordered = sorted(avg_rank, key=avg_rank.get)
    print(f"Average rank over {len(used)} value functions {used}:")
    for e in ordered:
        print(f"  {e:<15} {avg_rank[e]:.2f}")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    colors = [R.ODDSHAP_COLOR if e == "OddSHAP" else R.OKABE_ITO["sky"] for e in ordered] \
        if hasattr(R, "OKABE_ITO") else [R.estimator_style(e)["color"] for e in ordered]
    ax.barh(ordered, [avg_rank[e] for e in ordered], color=colors, edgecolor="black", lw=0.4)
    ax.invert_yaxis(); ax.set_xlabel("average rank (1 = best)")
    ax.set_title(R.fig_title("Table 1 — average rank", "all VFs", VARIANT, f"{len(used)} value functions"))
    R.add_banner(fig, BANNER_TAB); plt.show()

# %% [markdown]
# ## Table 3 — summary statistics (median, mean, **std**, Q1, Q3)
#
# The paper's Table 3 reports mean / median / Q1 / Q3 per estimator and value function. We
# reproduce that layout and **additionally report the standard deviation (std)** — the paper's
# table omits it, but it is requested for our write-up. One block of rows per estimator
# (OddSHAP first); one column per value function.

# %%
tbl = R.table3_dataframe(TAB_VFS + GPU_VFS, VARIANT)
try:
    display(tbl.style.set_caption(
        f"Table 3 — summary statistics of Shapley MSE (median / mean / std / Q1 / Q3) · "
        f"{R.VARIANT_LABEL[VARIANT]}"))
except (NameError, AttributeError):
    print(tbl)

# %% [markdown]
# ## Figure 2 — MSE vs budget (per value function)
#
# One **row per value function**, three panels each — all showing median ± IQR band, colours
# aligned with the paper (plus a redundant marker + line style, so the panels stay readable
# under colour vision deficiency):
#
# 1. **Ours** — our reproduction: every estimator we run, median line + IQR band.
# 2. **Paper** — the paper's original Figure-2 panel (its own figure, with the paper's IQR
#    bands and full method set including LeverageSHAP / 3-PolySHAP, which we do not run).
# 3. **OddSHAP: ours vs paper** — OddSHAP alone, our curve + IQR band (solid) over the paper's
#    median curve (dashed; the paper's per-curve band edges were not digitised), so the
#    reproduction fidelity is visible at a glance.
#
# > **kADDSHAP low-budget divergence (a finding, shown not hidden).** On the high-dimensional
# > value functions (d ≥ 60) the k-additive regression's normal-equations solve goes near-singular
# > at the smallest budgets and returns a diverged estimate (MSE up to 1e94, crime). We keep these
# > points: panel (1)'s y-axis is focused on the real estimates so the diverged line visibly shoots
# > off the top edge, with an annotation of the value. This is the regression-path sibling of the
# > TreeSHAP Chebyshev-Vandermonde instability we fixed upstream (PR #547) — `np.linalg.solve` on a
# > near-singular normal matrix returns garbage instead of raising. The paper's Figure 2 includes
# > neither kADDSHAP nor KernelSHAP — those are extra baselines we add.

# %%
import matplotlib.image as mpimg

for vf in TAB_VFS + GPU_VFS:                       # all 8 value functions
    ours = R.load_fig2(vf, VARIANT)
    paper = R.load_paper_fig2(vf)
    png = R.paper_figure_path(vf, "fig2")
    band = R.load_paper_oddshap_band(vf)           # digitised paper OddSHAP band, if available
    if not ours:
        continue
    is_gpu = vf in GPU_VFS
    banner = BANNER_GPU if is_gpu else BANNER_TAB
    # paper's own method key on top: the extracted per-VF paper panel has no legend of its
    # own (the paper shares one legend across sub-plots), and its colours match our panel (1),
    # so this single strip labels both panels 1 and 2.
    fig = plt.figure(figsize=(15, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 7], hspace=0.42)
    axl = fig.add_subplot(gs[0, :]); axl.axis("off")
    _leg = R.paper_legend_path()
    if _leg is not None:
        axl.imshow(mpimg.imread(str(_leg)))
        axl.set_title("paper's method key — same colours in panels (1) & (2)", fontsize=8)
    axes = [fig.add_subplot(gs[1, j]) for j in range(3)]

    # (1) ours — median + IQR band. Diverged points (kADDSHAP's near-singular low-budget
    # solve, MSE up to 1e94) are NOT hidden: they shoot off the top of the axis and are
    # annotated, because the divergence is itself a finding.
    ax = axes[0]
    finite_meds, diverged = [], []
    for e in ESTIMATORS:
        pts = sorted(ours.get(e, {}).items())
        if not pts:
            continue
        xs = [b for b, _ in pts]
        med = np.array([v[0] for _, v in pts])
        q1 = np.array([v[1] for _, v in pts]); q3 = np.array([v[2] for _, v in pts])
        st = R.paper_style(e)
        ax.plot(xs, np.clip(med, 1e-32, None), label=e, **st)
        ax.fill_between(xs, np.clip(q1, 1e-32, None), np.clip(q3, 1e-32, None),
                        color=st["color"], alpha=0.15)
        finite_meds += [m for m in med if np.isfinite(m) and m <= R.DIVERGED_MSE]
        diverged += [(e, b, m) for b, m in zip(xs, med) if m > R.DIVERGED_MSE]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    # focus the y-axis on the real estimates so the readable curves are not crushed by the
    # off-scale divergence; the diverged line still visibly shoots past the top edge.
    if finite_meds:
        ax.set_ylim(min(finite_meds) / 5, max(finite_meds) * 20)
    if diverged:
        w = max(diverged, key=lambda t: t[2])
        ax.annotate(f"{w[0]} diverges off-scale at m={w[1]}\n(near-singular solve, MSE≈{w[2]:.0e})",
                    xy=(0.03, 0.03), xycoords="axes fraction", va="bottom", fontsize=6.2,
                    color=R.paper_style(w[0])["color"],
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=R.paper_style(w[0])["color"], lw=0.6))
    ax.set_xlabel("budget m"); ax.set_ylabel("MSE (median ± IQR band)")
    ax.set_title("(1) ours (reproduction)"); ax.legend(fontsize=6, ncol=2)

    # (2) paper — the paper's original figure PNG (with IQR band) if we have it,
    # otherwise redrawn from the extracted median coordinates (no band digitised).
    ax = axes[1]
    if png is not None:
        ax.imshow(mpimg.imread(str(png))); ax.axis("off")
        ax.set_title("(2) paper (original Fig. 2, with IQR band)")
    elif paper:
        for m, pts in paper.items():
            ax.plot([b for b, _ in pts], [v for _, v in pts], label=m, **R.paper_style(m))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("budget m"); ax.legend(fontsize=6, ncol=2)
        ax.set_title("(2) paper (Fig. 2, extracted median)")
    else:
        ax.axis("off"); ax.set_title("(2) paper — figure not available")

    # (3) OddSHAP: ours (median + IQR band) vs paper (median, + band if digitised)
    ax = axes[2]
    op = sorted(ours.get("OddSHAP", {}).items())
    if op:
        xs = [b for b, _ in op]
        med = np.clip([v[0] for _, v in op], 1e-32, None)
        q1 = np.clip([v[1] for _, v in op], 1e-32, None); q3 = np.clip([v[2] for _, v in op], 1e-32, None)
        st = R.paper_style("OddSHAP")
        ax.plot(xs, med, label="OddSHAP (ours)", **st)
        ax.fill_between(xs, q1, q3, color=st["color"], alpha=0.18)
    if paper and "OddSHAP" in paper:
        pp = paper["OddSHAP"]
        pc = R.PAPER_COLOR["OddSHAP"]
        ax.plot([b for b, _ in pp], [v for _, v in pp], label="OddSHAP (paper, median)",
                color=pc, marker="x", linestyle="--", lw=1.8, ms=4)
        if band:  # digitised paper IQR band
            bx = [b for b, _, _, _ in band]
            bq1 = [q1 for _, _, q1, _ in band]; bq3 = [q3 for _, _, _, q3 in band]
            ax.fill_between(bx, bq1, bq3, color=pc, alpha=0.12, hatch="///", edgecolor=pc, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("budget m"); ax.set_ylabel("MSE (median ± IQR band)")
    ax.set_title("(3) OddSHAP: ours vs paper"); ax.legend(fontsize=7)

    fig.suptitle(R.fig_title("Figure 2 — MSE vs budget", vf, VARIANT), y=1.02)
    R.add_banner(fig, banner); plt.show()

# %% [markdown]
# ## Figure 4 & Figure 11 — interaction-sparsity (η) ablation
#
# MSE ratio vs the interaction-free baseline (median ± IQR band), at three fixed budgets
# (Figure 4 = 10,000; Figure 11 = 5,000 / 20,000). U-shape: adding odd interactions helps,
# then over-fits. Estate excluded (paper). Two panels per budget where the paper figure is
# available: **(1) ours** (each value function a distinct colour + marker + line style, with
# its IQR band) and **(2) the paper's original panel** (its own median ± IQR bands). Each
# value function's band shows the spread over the 30 instances.
#
# > **Baseline & colours.** The MSE ratio is normalised by the **interaction-free baseline**
# > (the paper's Figure 4 uses LeverageSHAP for this). The paper's Figure 4 *does* carry a
# > per-VF legend (DistilBERT, ViT16, Cancer, CG60, IL60, NHANES, Crime); our panel (1) colours
# > are aligned to it, except the paper's Cancer=red / ViT16=green pair is swapped to
# > colour-blind-safe hues (redundant markers keep every VF distinct). All value functions show
# > the same **U-shape** — odd interactions help, then over-fit — with Cancer lowest (≈1/100).

# %%
for budget, label in [(10_000, "Figure 4"), (5_000, "Figure 11a"), (20_000, "Figure 11c")]:
    png = R.paper_figure_path("ablation", "fig4") if budget == 10_000 else None
    ncol = 2 if png is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.4 * ncol, 4.4), squeeze=False)
    ax = axes[0][0]
    any_pts = False
    for vf in TAB_VFS + GPU_VFS:
        if vf == "realestate":
            continue
        pts = R.load_eta(vf, VARIANT, budget)
        if not pts:
            continue
        any_pts = True
        xs = [p[0] for p in pts]
        med = [p[1] for p in pts]; q1 = [p[2] for p in pts]; q3 = [p[3] for p in pts]
        st = R.paper_vf_style(vf)          # colours aligned to the paper's Fig. 4 legend
        ax.plot(xs, med, label=vf, **st)
        ax.fill_between(xs, q1, q3, color=st["color"], alpha=0.15)
    if not any_pts:
        plt.close(fig); continue
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="interaction-free baseline")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"number of odd interactions $\lceil m/\eta\rceil$")
    ax.set_ylabel("MSE ratio (median ± IQR band)")
    ax.set_title(f"(1) ours — {label}")
    ax.legend(fontsize=7, ncol=2)
    if png is not None:
        axp = axes[0][1]
        axp.imshow(mpimg.imread(str(png))); axp.axis("off")
        axp.set_title("(2) paper (original Fig. 4, with per-VF legend + IQR band)")
    fig.suptitle(R.fig_title(f"{label} — η ablation", "7 VFs", VARIANT, f"m={budget:,}"), y=1.02)
    R.add_banner(fig, BANNER_TAB); plt.show()

# %% [markdown]
# ## Figure 5 — runtime vs budget
#
# Wall-clock runtime (median) of each estimator vs budget. Reproduces the paper's claim
# that OddSHAP ≈ RegressionMSR in cost.

# %%
for vf in ["crime", "cancer"]:
    rt = R.load_runtime(vf, VARIANT)
    if not rt:
        continue
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for e in ESTIMATORS:
        pts = rt.get(e, [])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=e, **R.estimator_style(e))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("budget m"); ax.set_ylabel("runtime (s, median)")
    ax.set_title(R.fig_title("Figure 5 — runtime vs budget", vf, VARIANT))
    ax.legend(fontsize=7, ncol=2)
    R.add_banner(fig, BANNER_TAB); plt.show()

# %% [markdown]
# ## GPU value functions — ViT16, DistilBERT
#
# Deep-learning value functions with exact Shapley ground truth (2^16 / 2^14 evaluations).

# %%
for vf in GPU_VFS:
    t = R.load_table1(vf, VARIANT)
    if not t:
        continue
    print(f"\n{vf} (d={R.PAPER_D[vf]}, exact GT, budget=100·d):")
    for rk, e in enumerate(sorted(t, key=lambda e: t[e][0]), 1):
        m, q1, q3, _, _ = t[e]
        tag = "  <- OddSHAP" if e == "OddSHAP" else ""
        print(f"  {rk}. {e:<14} {m:.2e}  [IQR {q1:.1e}, {q3:.1e}]{tag}")

# %% [markdown]
# ## Summary
#
# Using this OddSHAP variant, the reproduction recovers the paper's headline across the
# value functions produced so far: OddSHAP's rank-1 average, the Figure-2 budget scaling,
# the Figure-4/11 interaction-sparsity U-shape, and the Figure-5 runtime profile. Each
# figure is annotated with the exact experimental environment that produced it, and uses a
# colour-blind-safe palette with redundant marker/line-style encoding.
