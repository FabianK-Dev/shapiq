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


# --- paper-aligned styling -------------------------------------------------- #
# The paper's own Figure-2 colours (extracted from the PDF vector layer), so our
# reproduction panel sits next to the paper panel with matching per-method colours.
# We keep a redundant marker + line style on top (the paper uses colour alone), so the
# figures remain readable under colour vision deficiency.
PAPER_COLOR = {
    "MSR": "#D0D0D0", "SVARM": "#8C8C8C", "PermutationSampling": "#555555", "PermSamp": "#555555",
    "ProxyLGBM": "#B0BEC5", "RegressionMSR": "#3F51B5",
    "3-PolySHAP": "#7CB342", "LeverageSHAP": "#00968A",
    "OddSHAP": "#E64A19",
    # our-only baselines (the paper's Fig. 2 has neither) — two CVD-safe hues the paper
    # does not use, so they never collide with a paper method. Without these both fell back
    # to the default grey and clashed with PermutationSampling.
    "KernelSHAP": "#CC79A7", "kADDSHAP": "#56B4E9",
}
# our-estimator name -> paper legend name (paper uses the full "PermutationSampling")
PAPER_METHOD_NAME = {"PermSamp": "PermutationSampling", "MSR": "MSR", "SVARM": "SVARM",
                     "RegressionMSR": "RegressionMSR", "OddSHAP": "OddSHAP"}
# our value-function id -> paper figure id (the paper uses short aliases)
PAPER_VF_ALIAS = {"realestate": "estate", "corrgroups60": "cg60",
                  "independentlinear60": "il60"}
_PAPER_MARKER = {"MSR": "o", "SVARM": "s", "PermSamp": "^", "PermutationSampling": "^",
                 "RegressionMSR": "P", "ProxyLGBM": "X", "3-PolySHAP": "*",
                 "LeverageSHAP": "D", "OddSHAP": "o", "KernelSHAP": "D", "kADDSHAP": "v"}
_PAPER_LINESTYLE = {"MSR": ":", "SVARM": "--", "PermSamp": "-.", "PermutationSampling": "-.",
                    "RegressionMSR": "-", "ProxyLGBM": ":", "3-PolySHAP": "--",
                    "LeverageSHAP": "-.", "OddSHAP": "-", "KernelSHAP": "--", "kADDSHAP": ":"}


# Paper Figure-4 per-value-function colours, aligned to the paper's own legend so our
# ablation panel sits line-for-line beside the paper's. The paper uses Cancer=red and
# ViT16=green (a red-green pair the user cannot separate), so those two are swapped to
# CVD-safe hues; every VF also keeps a distinct marker + line style.
PAPER_VF_STYLE = {
    "distilbert":          dict(color="#0072B2", marker="o", linestyle="-"),   # paper blue (match)
    "vit16":               dict(color="#56B4E9", marker="s", linestyle="--"),  # paper green → sky
    "cancer":              dict(color="#D55E00", marker="^", linestyle="-"),   # paper red → vermillion
    "corrgroups60":        dict(color="#CC79A7", marker="D", linestyle="-."),  # paper purple (match)
    "independentlinear60": dict(color="#E69F00", marker="v", linestyle=":"),   # paper brown → orange
    "nhanes":              dict(color="#000000", marker="P", linestyle="-"),   # paper pink → black
    "crime":               dict(color="#999999", marker="X", linestyle="--"),  # paper grey (match)
}


def paper_vf_style(vf: str) -> dict:
    """Paper-Figure-4-aligned plot kwargs for a value-function line (CVD-safe swaps)."""
    base = dict(PAPER_VF_STYLE.get(vf, dict(color="#000000", marker="o", linestyle="-")))
    base.update(lw=1.9, ms=4)
    return base


def paper_style(method: str, *, emphasize: bool = None) -> dict:
    """Paper-aligned plot kwargs (colour from the paper, plus redundant marker/line style)."""
    emph = (method == "OddSHAP") if emphasize is None else emphasize
    return dict(color=PAPER_COLOR.get(method, "#555555"),
                marker=_PAPER_MARKER.get(method, "o"),
                linestyle=_PAPER_LINESTYLE.get(method, "-"),
                lw=2.6 if emph else 1.3, ms=4 if emph else 3)
