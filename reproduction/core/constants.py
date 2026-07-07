"""Single source of truth for the reproduction: estimator names, ablation grids,
value-function metadata, variant tags, and the CSV schemas. Every other module imports
from here so a change lands in exactly one place.
"""

from __future__ import annotations

# --- estimators ------------------------------------------------------------- #
BASELINE_ESTIMATORS = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR"]
ESTIMATORS = [*BASELINE_ESTIMATORS, "OddSHAP"]

# --- ablation grids (paper Section 5.3 / Appendix D.3) ---------------------- #
ETAS = [50, 10, 5, 2]
ETA_BUDGETS = [5_000, 10_000, 20_000]   # Figure 4 (10000) + Figure 11 (5000 / 20000)
INTERACTION_FREE_FACTOR = 10            # arbitrary eta for the interaction-free baseline
                                        # (its higher-order support is emptied anyway)

# --- OddSHAP variants ------------------------------------------------------- #
DEFAULT_VARIANT = "v522_merged"
VARIANT_CHOICES = ["v522_merged", "v560_improved", "library"]
VARIANT_LABEL = {
    "v522_merged": "OddSHAP PR #522 (ours, merged)",
    "v560_improved": "OddSHAP PR #560 (author improvement)",
    "library": "installed shapiq.OddSHAP",
}
VARIANT_SHORT = {"v522_merged": "#522", "v560_improved": "#560", "library": "lib"}

# --- value functions -------------------------------------------------------- #
# paper feature dimension d per value function (used in titles / n*eta budgets)
PAPER_D = {"cancer": 30, "realestate": 15, "corrgroups60": 60, "independentlinear60": 60,
           "nhanes": 79, "crime": 101, "vit16": 16, "distilbert": 14}
TABULAR_VF_NAMES = ["cancer", "realestate", "corrgroups60", "independentlinear60", "nhanes", "crime"]
GPU_VF_NAMES = ["vit16", "distilbert"]

# --- CSV schemas (writers and readers both import these) -------------------- #
SCHEMA_TABLE1 = ["vf", "estimator", "variant", "n", "budget", "median", "q1", "q3", "mean", "std"]
SCHEMA_FIG2 = ["vf", "estimator", "variant", "budget", "n", "median", "q1", "q3"]
SCHEMA_ETA = ["vf", "variant", "budget", "eta", "n_interactions", "n", "median_mse", "ratio_vs_base"]
SCHEMA_RUNTIME = ["vf", "estimator", "variant", "budget", "n", "median_runtime_s"]
