"""Colour-blind-safe plotting styles for the reproduction figures.

The user has a red-green colour vision deficiency, so **no figure may use red/green as a
sole encoding**. We use the Okabe-Ito qualitative palette (designed for all common CVD
types) and, critically, give every series a **redundant non-colour cue** — a distinct
marker shape *and* line style — so the figures also read in greyscale and for any CVD.

One entity = one style, fixed across every notebook: import ``estimator_style`` /
``vf_style`` and never fall back to matplotlib's default colour cycle.
"""

from __future__ import annotations

from .constants import ESTIMATORS, PAPER_D

# Okabe-Ito (2008) — 8 colours distinguishable under protanopia/deuteranopia/tritanopia.
OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
    "grey": "#999999",
}

# OddSHAP gets the single most salient colour (vermillion) everywhere — one role, one hex.
ODDSHAP_COLOR = OKABE_ITO["vermillion"]

# per-estimator: colour + marker + line style (three redundant channels). No two share a
# colour; green is avoided so it never pairs with OddSHAP's vermillion (a red-green risk).
# RegressionMSR — OddSHAP's closest rival — is black, maximally distinct from vermillion.
_ESTIMATOR_STYLE = {
    "MSR":           dict(color=OKABE_ITO["grey"],   marker="o", linestyle=":"),
    "SVARM":         dict(color=OKABE_ITO["sky"],    marker="s", linestyle="--"),
    "PermSamp":      dict(color=OKABE_ITO["blue"],   marker="^", linestyle="-."),
    "KernelSHAP":    dict(color=OKABE_ITO["purple"], marker="D", linestyle="--"),
    "kADDSHAP":      dict(color=OKABE_ITO["orange"], marker="v", linestyle=":"),
    "RegressionMSR": dict(color=OKABE_ITO["black"],  marker="P", linestyle="-"),
    "OddSHAP":       dict(color=ODDSHAP_COLOR,       marker="o", linestyle="-"),
}

# variant comparison (#522 vs #560) — a CVD-safe pair with redundant line style
VARIANT_STYLE = {
    "v522_merged":   dict(color=OKABE_ITO["vermillion"], marker="o", linestyle="-"),
    "v560_improved": dict(color=OKABE_ITO["blue"],       marker="s", linestyle="--"),
    "library":       dict(color=OKABE_ITO["black"],      marker="x", linestyle=":"),
}

# per value-function styles (for the multi-line eta ablation) — Okabe-Ito + distinct
# markers/line styles. Green is omitted so no line pairs green with vermillion.
_VF_COLORS = [OKABE_ITO["vermillion"], OKABE_ITO["blue"], OKABE_ITO["orange"],
              OKABE_ITO["sky"], OKABE_ITO["purple"], OKABE_ITO["black"],
              OKABE_ITO["yellow"], OKABE_ITO["grey"]]
_VF_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_VF_LINES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
_ALL_VFS = [*PAPER_D]  # stable ordering


def estimator_style(name: str, *, emphasize: bool = None) -> dict:
    """Return plot kwargs (color/marker/linestyle/lw/ms) for an estimator.

    OddSHAP is emphasised (thicker) by default; pass ``emphasize`` to override.
    """
    base = dict(_ESTIMATOR_STYLE.get(name, dict(color=OKABE_ITO["black"], marker="o", linestyle="-")))
    emph = (name == "OddSHAP") if emphasize is None else emphasize
    base.update(lw=2.6 if emph else 1.3, ms=4 if emph else 3)
    return base


def vf_style(vf: str) -> dict:
    """Return plot kwargs for a value-function line (colour + marker + line style)."""
    i = _ALL_VFS.index(vf) if vf in _ALL_VFS else 0
    return dict(color=_VF_COLORS[i % len(_VF_COLORS)], marker=_VF_MARKERS[i % len(_VF_MARKERS)],
                linestyle=_VF_LINES[i % len(_VF_LINES)], lw=1.8, ms=4)


def variant_style(variant: str, *, emphasize: bool = False) -> dict:
    base = dict(VARIANT_STYLE.get(variant, VARIANT_STYLE["library"]))
    base.update(lw=2.4 if emphasize else 1.8, ms=4)
    return base
