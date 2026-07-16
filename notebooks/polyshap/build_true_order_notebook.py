"""Generate polyshap_true_order.ipynb.

A standalone reproduction-style notebook (no side-by-side paper panels) that
shows the paper's central claim on a game with a *known* true interaction order:
PolySHAP improves on KernelSHAP as ``max_order`` rises **up to the game's true
interaction order k**, and overshooting k only costs budget.

Game: a **Sum of Unanimity Models (SOUM)** whose ``max_interaction_size`` *is*
the true interaction order k (its Moebius representation has no terms above
order k). One game to start with; the layout is a template (add entries to
``GAME_SPECS`` to expand).

Focus:
* ``max_order = k-1`` (below), ``= k`` (match), ``= k+1`` (overshoot), plus
  KernelSHAP (order 1) as the blue baseline.
* A **dense budget grid near 2**n** so the *late, non-linear collapse* of the
  under-order methods to the exact Shapley value (only at full enumeration) is
  shown honestly, not as a straight interpolation to the final point.

Run ``python notebooks/polyshap/build_true_order_notebook.py`` to (re)write it.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# PolySHAP vs. the true interaction order (SOUM games)

This notebook reproduces the paper's central figure/table behaviour
(Fumagalli et al., *PolySHAP*, ICLR 2026) on a game whose **true interaction
order is known by construction**, so "below / matching / overshooting the true
order" is unambiguous.

**Game.** A *Sum of Unanimity Models* (`SOUM`): a weighted sum of unanimity
games, each supported on a coalition of size ≤ `max_interaction_size`. Its
Möbius (interaction) representation therefore has **no terms above
`max_interaction_size`** — that value *is* the game's **true interaction order
`k`**. Ground-truth Shapley values come from `shapiq.ExactComputer`.

**What we show.** With `k` fixed, we sweep `PolySHAPKAdd(max_order=…)` at
`k-1` (below), `k` (match) and `k+1` (overshoot), plus `KernelSHAP` (= order 1)
as a blue baseline:

* `max_order < k` — **underfits**: error stays above the matched fit at every
  affordable budget.
* `max_order = k` — **matches**: the surrogate represents the game exactly, so
  the estimate collapses to the true Shapley values as soon as the budget
  covers the frontier.
* `max_order = k+1` — **overshoots**: same accuracy as the match, but only once
  a *larger* frontier is affordable → wasted budget, never a gain.

**The late collapse.** KernelSHAP (and any `max_order < k`) only becomes exact
at **full enumeration** (`2**n`). We use a **dense budget grid near `2**n`** so
this collapse is shown as it truly is — flat/elevated until the very end, then a
sharp drop — rather than the misleading straight line a coarse grid draws from
the second-to-last point to the exact final point.

One game is set up here as a template; add rows to `GAME_SPECS` to expand.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 0 · Setup

Self-contained: only `numpy`/`matplotlib`, the integrated `shapiq`
approximators, and the `SOUM` benchmark game. Sweeps are cached under `cache/`;
figures are written to `plots/`.
""")

code(r"""
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shapiq
from shapiq import ExactComputer
from shapiq.approximator import KernelSHAP
from shapiq.approximator.regression.polyshap import PolySHAPKAdd
from shapiq_games.synthetic import SOUM

HERE = Path.cwd()
NB_DIR = next((c for c in (HERE, HERE / "polyshap", HERE / "notebooks" / "polyshap")
               if (c / "build_true_order_notebook.py").exists()), HERE)
CACHE_DIR = NB_DIR / "cache"; CACHE_DIR.mkdir(exist_ok=True)
PLOTS_DIR = NB_DIR / "plots"; PLOTS_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path

print("integrated shapiq:", shapiq.__file__)
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1 · Colour scheme and roles

Same colour-per-order scheme as the other PolySHAP notebooks (teal→amber→deep
orange→dark red for orders 1–4), with **`KernelSHAP` in blue** as the baseline.
Here the three PolySHAP curves are labelled by their **role relative to the true
order `k`**: below (`k-1`), match (`k`), overshoot (`k+1`).
""")

code(r"""
COLORS = {
    "KernelSHAP": "#1f77b4",   # blue  — baseline (= order 1)
    "below":      "#ffb64d",   # amber       (max_order = k-1)
    "match":      "#e64918",   # deep orange (max_order = k)
    "over":       "#bf360b",   # dark red    (max_order = k+1)
}
MARKER = {"KernelSHAP": "o", "below": "s", "match": "^", "over": "D"}
LINESTYLE = {"KernelSHAP": "--", "below": "-", "match": "-", "over": "-"}
""")

# --------------------------------------------------------------------------- #
md(r"""
## 2 · The game and its true interaction order

`SOUM(n, n_basis_games, max_interaction_size=k)` builds a game whose true
interaction order is exactly `k = max_interaction_size`. We start with one
game (`SOUM-12`, `k=3`); average over a few random instances for smooth curves.
""")

code(r"""
@dataclass(frozen=True)
class GameSpec:
    name: str
    n: int
    k: int              # true interaction order == SOUM max_interaction_size
    n_basis_games: int

# One game to start (template). Add more entries to expand the study.
GAME_SPECS = {
    "SOUM-12": GameSpec("SOUM-12", n=12, k=3, n_basis_games=30),
}

N_INSTANCES = 5     # random game instances averaged per point (mean ± SEM)
SEED = 40

def make_game(spec: GameSpec, seed: int):
    return SOUM(n=spec.n, n_basis_games=spec.n_basis_games,
               max_interaction_size=spec.k, min_interaction_size=1,
               random_state=seed)

def exact_shapley(game, n: int) -> np.ndarray:
    sv = ExactComputer(game, n_players=n)(index="SV", order=1)
    return np.asarray(sv.get_n_order_values(1), dtype=float)
""")

# --------------------------------------------------------------------------- #
md(r"""
## 3 · Estimators, roles, and the (dense) budget grid

The three PolySHAP roles derive from the true order `k`: `max_order = k-1, k,
k+1`. `PolySHAPKAdd(max_order=m)` needs at least `n_variables = 1 + C(n,1) + … +
C(n,m)` evaluations, so higher orders only start further to the right.

The budget grid is **log-spaced and additionally densified near `2**n`** so the
late collapse of the under-order methods is resolved rather than interpolated.
""")

code(r"""
def roles_for(spec: GameSpec) -> dict[str, int]:
    # role -> PolySHAP max_order
    return {"below": spec.k - 1, "match": spec.k, "over": spec.k + 1}

def make_method(role: str, spec: GameSpec, random_state: int):
    weights = np.ones(spec.n + 1)   # uniform over subset sizes (order-1 leverage)
    if role == "KernelSHAP":
        return KernelSHAP(n=spec.n, sampling_weights=weights, random_state=random_state)
    return PolySHAPKAdd(n=spec.n, max_order=roles_for(spec)[role],
                        sampling_weights=weights, random_state=random_state)

def budget_grid(n: int) -> list[int]:
    lo, hi = n + 1, 2 ** n
    base = np.logspace(np.log10(lo), np.log10(hi), 22)
    # extra density near full enumeration (2**n) to reveal the late collapse
    near = np.array([0.80, 0.88, 0.93, 0.96, 0.98, 0.99, 0.995, 0.999, 1.0]) * hi
    grid = np.unique(np.round(np.concatenate([base, near])).astype(int))
    grid = grid[(grid >= lo) & (grid <= hi)]
    return sorted(int(b) for b in grid)

ROLES = ["KernelSHAP", "below", "match", "over"]

def n_variables(role: str, spec: GameSpec) -> int:
    return getattr(make_method(role, spec, 0), "n_variables", spec.n)
""")

code(r"""
import hashlib, json

def run_sweep() -> pd.DataFrame:
    rows = []
    for spec in GAME_SPECS.values():
        budgets = budget_grid(spec.n)
        for inst in range(1, N_INSTANCES + 1):
            game = make_game(spec, seed=SEED + inst)
            exact = exact_shapley(game, spec.n)
            for budget in budgets:
                for role in ROLES:
                    est_obj = make_method(role, spec, SEED + inst)
                    if getattr(est_obj, "n_variables", 0) > budget:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        iv = est_obj.approximate(budget=budget, game=game)
                    est = np.asarray(iv.get_n_order_values(1), dtype=float)
                    rows.append({"game": spec.name, "n": spec.n, "k": spec.k,
                                 "instance": inst, "budget": budget, "role": role,
                                 "MSE": float(np.mean((est - exact) ** 2))})
    return pd.DataFrame(rows)

def cached_sweep() -> pd.DataFrame:
    key = {"games": {n: (s.n, s.k, s.n_basis_games) for n, s in GAME_SPECS.items()},
           "N": N_INSTANCES, "seed": SEED}
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    csv = CACHE_DIR / f"true_order_{digest}.csv"
    if csv.exists():
        print("[cache hit ]", csv.name); return pd.read_csv(csv)
    print("[cache miss] computing ->", csv.name)
    df = run_sweep(); df.to_csv(csv, index=False)
    return df

df = cached_sweep()
print(f"{len(df)} cells over {N_INSTANCES} instance(s)")
df.head()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 4 · Figure — error vs. budget

Log–log axes, mean ± SEM across instances. Dotted vertical guides mark where the
degree-`k` (match) and degree-`k+1` (overshoot) frontiers first become
affordable (`n_variables`); the dashed vertical line marks **full enumeration**
(`2**n`), where every method is exact. Note how KernelSHAP / below stay elevated
almost all the way to `2**n` and only then drop — the collapse is late and
sharp, not linear.
""")

code(r"""
MSE_FLOOR = 1e-13   # log-axis display floor; exact fits (and the 2**n point) clamp here

def agg(df):
    g = df.groupby(["game", "role", "budget"])["MSE"]
    return g.agg(["mean", "sem"]).reset_index()

def panel(ax, A, spec):
    sub = A[A["game"] == spec.name]
    for role in ROLES:
        s = sub[sub["role"] == role].sort_values("budget")
        if s.empty:
            continue
        mean = np.maximum(s["mean"].to_numpy(), MSE_FLOOR)
        sem = np.nan_to_num(s["sem"].to_numpy())
        m_o = roles_for(spec)
        lbl = {"KernelSHAP": "KernelSHAP (order 1)",
               "below": f"max_order={m_o['below']} (k−1, below)",
               "match": f"max_order={m_o['match']} (k, match)",
               "over":  f"max_order={m_o['over']} (k+1, overshoot)"}[role]
        ax.plot(s["budget"], mean, ls=LINESTYLE[role], marker=MARKER[role], ms=5,
                lw=1.8, color=COLORS[role], label=lbl)
        ax.fill_between(s["budget"], np.maximum(mean - sem, MSE_FLOOR), mean + sem,
                        color=COLORS[role], alpha=0.15, lw=0)
    # frontier-affordability guides + full-enumeration line
    ax.axvline(n_variables("match", spec), color=COLORS["match"], ls=":", lw=1.2, alpha=0.7)
    ax.axvline(n_variables("over", spec), color=COLORS["over"], ls=":", lw=1.2, alpha=0.7)
    ax.axvline(2 ** spec.n, color="0.4", ls="--", lw=1.2, alpha=0.8)
    ax.text(2 ** spec.n, ax.get_ylim()[1], " 2ⁿ (exact)", color="0.4",
            va="top", ha="right", rotation=90, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(MSE_FLOOR * 0.5, None)
    ax.set_xlabel("Budget (coalitions evaluated)")
    ax.set_title(f"{spec.name} — SOUM, true interaction order k={spec.k}")
    ax.grid(True, which="both", ls=":", alpha=0.35)

A = agg(df)
fig, axes = plt.subplots(1, len(GAME_SPECS), figsize=(6.4 * len(GAME_SPECS), 4.6),
                         squeeze=False)
for ax, spec in zip(axes[0], GAME_SPECS.values()):
    panel(ax, A, spec)
axes[0][0].set_ylabel("MSE vs. exact Shapley (± SEM)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), frameon=False)
fig.suptitle("PolySHAP: below / matching / overshooting the true interaction order",
             y=1.005, fontsize=12)
fig.tight_layout()
save_fig(fig, "true_order_SOUM.png")
plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
### Reading the figure

* **`max_order = k` (match, deep orange)** drops to the floor as soon as the
  budget reaches its frontier size (left dotted guide) — the surrogate captures
  the game exactly.
* **`max_order = k+1` (overshoot, dark red)** reaches the same floor, but only
  from the *right* dotted guide — a larger budget for **no** accuracy gain.
* **`max_order = k-1` (below, amber)** and **KernelSHAP (blue)** never capture
  the full interaction structure: their error decays slowly and only collapses
  to exact at the dashed **`2**n`** line. The dense sampling there shows this
  collapse is **late and sharp**, not the straight line a coarse grid implies.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 5 · Table — MSE at two reference budgets

At a **mid budget** (smallest where all four are affordable) and the **largest
sub-full budget** (just below `2**n`), mean MSE should read
below > KernelSHAP-ish > match ≈ over, with match/over already at the floor and
the under-order methods still clearly non-zero just before full enumeration.
""")

code(r"""
def table():
    out = {}
    for spec in GAME_SPECS.values():
        grid = budget_grid(spec.n)
        mid = min(b for b in grid if b >= n_variables("over", spec))
        near_full = max(b for b in grid if b < 2 ** spec.n)
        for label, b in [(f"mid (m={mid})", mid), (f"near-full (m={near_full})", near_full)]:
            sub = df[(df["game"] == spec.name) & (df["budget"] == b)]
            out[f"{spec.name} · {label}"] = sub.groupby("role")["MSE"].mean()
    tab = pd.DataFrame(out).reindex(ROLES)
    tab.index = ["KernelSHAP (o1)", "below (k−1)", "match (k)", "over (k+1)"]
    return tab

with pd.option_context("display.float_format", lambda v: f"{v:.2e}"):
    print(table())
table()
""")

# --------------------------------------------------------------------------- #
md(r"""
## Conclusion & how to extend

On a SOUM game with a *known* true interaction order `k`, PolySHAP behaves
exactly as the paper predicts: accuracy improves as `max_order` rises **towards
`k`**, is **best (exact)** at `max_order = k`, and **gains nothing** beyond `k`
while demanding a larger budget. KernelSHAP and any `max_order < k` only reach
the exact Shapley values at full enumeration — a **late, sharp** collapse that
the dense budget grid makes visible.

**Template.** To expand the study, add entries to `GAME_SPECS` (e.g. other `n`,
other true orders `k`, or a different game family) — the sweep, figure, and
table loop over `GAME_SPECS` unchanged.
""")

# --------------------------------------------------------------------------- #
NB["cells"] = cells
NB["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent / "polyshap_true_order.ipynb"
nbf.write(NB, str(out))
print("wrote", out)
