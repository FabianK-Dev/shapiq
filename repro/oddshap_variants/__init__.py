"""Pluggable OddSHAP implementation registry.

The reproduction harness can run against *different* OddSHAP implementations so we
can compare, on identical value functions / budgets / seeds, how the estimator's
numbers move between revisions. Each variant is a vendored, standalone copy of one
`oddshap.py` revision (its imports are all stable public shapiq API, so the copies
coexist in one process).

Variants
--------
- ``v522_merged``   : the implementation merged into upstream via PR #522 (our group's
                      contribution, pre-#560). Minimum budget ``n * interaction_factor``;
                      candidate count ``ceil(m/eta) - d``; the full paper Algorithm 1.
- ``v560_improved`` : the paper author's follow-up PR #560. Minimum budget relaxed to
                      ``interaction_factor``; the surrogate tree also selects the most
                      important individuals in the low-budget regime; paired rows merged.
- ``library``       : whatever ``shapiq.approximator.OddSHAP`` currently resolves to
                      (the installed version — sanity anchor).

Add a new PR/revision by dropping its ``oddshap.py`` into this folder as
``_oddshap_<tag>.py`` and registering it in ``VARIANTS`` below. Nothing else changes.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# name -> (module basename, human label)
_VARIANT_MODULES: dict[str, tuple[str, str]] = {
    "v522_merged": ("_oddshap_v522_merged", "PR #522 (merged, ours)"),
    "v560_improved": ("_oddshap_v560_improved", "PR #560 (author improvement)"),
}


def available_variants() -> list[str]:
    """Return the registered variant names (plus the installed-library anchor)."""
    return [*_VARIANT_MODULES, "library"]


def variant_label(name: str) -> str:
    """Human-readable label for a variant name."""
    if name == "library":
        return "installed shapiq.OddSHAP"
    return _VARIANT_MODULES[name][1]


def load_oddshap(variant: str) -> Callable:
    """Return the ``OddSHAP`` class for a registered variant.

    ``variant="library"`` returns the installed ``shapiq.approximator.OddSHAP``; any
    other name loads the vendored standalone copy of that revision.
    """
    if variant == "library":
        from shapiq.approximator import OddSHAP

        return OddSHAP
    if variant not in _VARIANT_MODULES:
        msg = f"unknown OddSHAP variant {variant!r}; available: {available_variants()}"
        raise KeyError(msg)
    module_name, _ = _VARIANT_MODULES[variant]
    module = importlib.import_module(f"{__name__}.{module_name}")
    return module.OddSHAP
