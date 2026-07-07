# %% [markdown]
# # NB1 — OddSHAP paper reproduction with **our merged contribution (PR #522)**
#
# This notebook reproduces the paper's results (Fumagalli et al. 2026, *An Odd Estimator
# for Shapley Values*, arXiv:2602.01399) using the OddSHAP implementation **our group
# contributed and merged (PR #522)**. Every figure names its value function, dimension,
# the OddSHAP variant, and carries an experimental-environment banner.
#
# **Provenance.** All numbers are produced by the cluster scripts in `reproduction/cluster/`
# (variant `v522_merged`) and written to the gitignored `reproduction/data/` folder; this
# notebook reads them and plots. Regenerate everything with
# `bash reproduction/cluster/submit_all.sh v522_merged`.
#
# **Value function** (paper Table 2): XGBoost + interventional perturbation, 50 background
# samples; ground truth = exact interventional TreeSHAP (tabular) / exact enumeration (GPU).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path.cwd() if Path("reproduction").is_dir() else Path.cwd().parent))
from reproduction.core import report as R

VARIANT = "v522_merged"
TAB_VFS = ["cancer", "realestate", "corrgroups60", "independentlinear60", "nhanes", "crime"]
GPU_VFS = ["vit16", "distilbert"]
BANNER_TAB = R.environment_banner(VARIANT, gt="exact interventional TreeSHAP")
BANNER_GPU = R.environment_banner(VARIANT, gt="exact Shapley (2^d enumeration)",
                                  vf_family="deep model (ViT16 / DistilBERT), baseline imputation")
print("Missing data (run the cluster scripts to fill):",
      [vf for vf in TAB_VFS + GPU_VFS if not R.has(f"table1_{vf}_{VARIANT}.csv")] or "none")

# %% [markdown]
# ## Table 1 & Table 3 — average MSE, rank, and spread
#
# Median MSE at budget ≈ 100·d, with Q1/Q3 and std. OddSHAP's average rank across value
# functions is the paper's headline claim.

# %%
avg_rank, used = R.average_rank(TAB_VFS + GPU_VFS, VARIANT)
if avg_rank:
    print(f"Average rank over {len(used)} value functions {used}:")
    for e in sorted(avg_rank, key=avg_rank.get):
        print(f"  {e:<15} {avg_rank[e]:.2f}")
    ordered = sorted(avg_rank, key=avg_rank.get)
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.barh(ordered, [avg_rank[e] for e in ordered],
            color=["#CC3311" if e == "OddSHAP" else "#88AACC" for e in ordered], edgecolor="black", lw=0.4)
    ax.invert_yaxis(); ax.set_xlabel("average rank (1 = best)")
    ax.set_title(R.fig_title("Table 1 — average rank", "all VFs", VARIANT, f"{len(used)} value functions"))
    R.add_banner(fig, BANNER_TAB); fig.tight_layout(); plt.show()

# Table 3 — expanded, per VF
for vf in TAB_VFS + GPU_VFS:
    t = R.load_table1(vf, VARIANT)
    if not t:
        continue
    print(f"\n{vf} (d={R.PAPER_D[vf]}) — median [Q1, Q3]  (std):")
    for e in sorted(t, key=lambda e: t[e][0]):
        m, q1, q3, mean, std = t[e]
        tag = "  <- OddSHAP" if e == "OddSHAP" else ""
        print(f"  {e:<15} {m:.2e}  [{q1:.2e}, {q3:.2e}]  (std {std:.1e}){tag}")

# %% [markdown]
# ## Figure 2 — MSE vs budget (per value function)
#
# OddSHAP (red) vs the sampling baselines. Log-log; median line, IQR band. OddSHAP turns
# on at budget ≥ n·η and then drops fastest.

# %%
for vf in TAB_VFS:
    curves = R.load_fig2(vf, VARIANT)
    if not curves:
        continue
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for e in R.ESTIMATORS:
        pts = sorted(curves.get(e, {}).items())
        if not pts:
            continue
        xs = [b for b, _ in pts]
        med = np.array([v[0] for _, v in pts])
        q1 = np.array([v[1] for _, v in pts]); q3 = np.array([v[2] for _, v in pts])
        col = R.PCOL.get(e, (.5, .5, .5))
        ax.plot(xs, np.clip(med, 1e-32, None), marker="o", ms=3, color=col,
                lw=2.4 if e == "OddSHAP" else 1.2, label=e)
        ax.fill_between(xs, np.clip(q1, 1e-32, None), np.clip(q3, 1e-32, None), color=col, alpha=0.12)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("budget m"); ax.set_ylabel("Shapley MSE (median, IQR band)")
    ax.set_title(R.fig_title("Figure 2 — MSE vs budget", vf, VARIANT))
    ax.legend(fontsize=7, ncol=2)
    R.add_banner(fig, BANNER_TAB); fig.tight_layout(); plt.show()

# %% [markdown]
# ## Figure 4 & Figure 11 — interaction-sparsity (η) ablation
#
# MSE ratio vs the interaction-free baseline, at three fixed budgets (Figure 4 = 10,000;
# Figure 11 = 5,000 / 20,000). The U-shape: adding odd interactions helps, then over-fits.
# Estate is excluded (paper).

# %%
for budget, label in [(10_000, "Figure 4"), (5_000, "Figure 11a"), (20_000, "Figure 11c")]:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    any_pts = False
    for vf in TAB_VFS + GPU_VFS:
        if vf == "realestate":
            continue
        pts = R.load_eta(vf, VARIANT, budget)
        if not pts:
            continue
        any_pts = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=vf)
    if not any_pts:
        plt.close(fig); continue
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="interaction-free baseline")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"number of odd interactions $\lceil m/\eta\rceil$")
    ax.set_ylabel("MSE ratio vs interaction-free")
    ax.set_title(R.fig_title(f"{label} — η ablation", "7 VFs", VARIANT, f"m={budget:,}"))
    ax.legend(fontsize=8, ncol=2)
    R.add_banner(fig, BANNER_TAB); fig.tight_layout(); plt.show()

# %% [markdown]
# ## Figure 5 — runtime vs budget
#
# Wall-clock runtime (median) of each estimator vs budget. Reproduces the paper's claim
# that OddSHAP ≈ RegressionMSR in cost, both above the interaction-free baseline at high m.

# %%
for vf in ["crime", "cancer"]:
    name = f"runtime_{vf}_{VARIANT}.csv"
    if not R.has(name):
        continue
    rt = {}
    for r in R.read(name):
        rt.setdefault(r["estimator"], []).append((int(r["budget"]), float(r["median_runtime_s"])))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for e in R.ESTIMATORS:
        pts = sorted(rt.get(e, []))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3,
                    color=R.PCOL.get(e, (.5, .5, .5)), lw=2.4 if e == "OddSHAP" else 1.2, label=e)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("budget m"); ax.set_ylabel("runtime (s, median)")
    ax.set_title(R.fig_title("Figure 5 — runtime vs budget", vf, VARIANT))
    ax.legend(fontsize=7, ncol=2)
    R.add_banner(fig, BANNER_TAB); fig.tight_layout(); plt.show()

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
# Using our merged PR #522 OddSHAP, the reproduction recovers the paper's headline across
# the value functions produced so far: OddSHAP's rank-1 average, the Figure-2 budget
# scaling, and the Figure-4/11 interaction-sparsity U-shape. Each figure above is annotated
# with the exact experimental environment that produced it.
