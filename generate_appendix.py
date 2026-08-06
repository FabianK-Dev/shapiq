from __future__ import annotations

import html
import math
from pathlib import Path

from PIL import Image as PILImage
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ==============================================================================
# 1. INPUT FILE LIST (PASTE YOUR GIT DIFF OUTPUT HERE)
# ==============================================================================
RAW_GIT_DIFF = """
.gitattributes
.github/workflows/ci.yml
.gitignore
CHANGELOG.md
Pr_S26 - Group G - New Shapley Value Approximators Submission.zip
SUBMISSION_README.md
benchmark/README.md
benchmark/__init__.py
benchmark/_discovery.py
benchmark/performance.py
benchmark/results/lmu_full_sweep_20260717/README.md
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/KendallTau_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/L2_Norm_Error_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/MAE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/MSE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_10_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/Precision_at_5_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/SAE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/SSE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots/runtime_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/KendallTau_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/L2_Norm_Error_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MAE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/MSE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_10_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/Precision_at_5_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/README.md
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SAE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/SSE_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_Adult_n12.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_California_n8.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_Communities_n101.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_Correlated_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_Diabetes_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_IRIS_n4.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_Independent_n60.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_NHANES_n79.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_SOUM_n10.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_SOUM_n6.png
benchmark/results/lmu_full_sweep_20260717/plots_paper_subset/runtime_SOUM_n8.png
benchmark/results/lmu_full_sweep_20260717/results.csv
benchmark/results/sv_sweep_20260717_004219/plots/KendallTau_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/KendallTau_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/KendallTau_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/KendallTau_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/KendallTau_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/KendallTau_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/KendallTau_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/L2_Norm_Error_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/L2_Norm_Error_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/L2_Norm_Error_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/L2_Norm_Error_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/L2_Norm_Error_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/L2_Norm_Error_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/L2_Norm_Error_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/MAE_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/MAE_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/MAE_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/MAE_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/MAE_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/MAE_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/MAE_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/MSE_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/MSE_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/MSE_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/MSE_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/MSE_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/MSE_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/MSE_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_10_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_10_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_10_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_10_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_10_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_10_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_10_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_5_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_5_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_5_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_5_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_5_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_5_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/Precision_at_5_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/SAE_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/SAE_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/SAE_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/SAE_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/SAE_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/SAE_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/SAE_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/SSE_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/SSE_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/SSE_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/SSE_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/SSE_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/SSE_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/SSE_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/runtime_Adult_n12.png
benchmark/results/sv_sweep_20260717_004219/plots/runtime_California_n8.png
benchmark/results/sv_sweep_20260717_004219/plots/runtime_Diabetes_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/runtime_IRIS_n4.png
benchmark/results/sv_sweep_20260717_004219/plots/runtime_SOUM_n10.png
benchmark/results/sv_sweep_20260717_004219/plots/runtime_SOUM_n6.png
benchmark/results/sv_sweep_20260717_004219/plots/runtime_SOUM_n8.png
benchmark/results/sv_sweep_20260717_004219/results.csv
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/KendallTau_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/L2_Norm_Error_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/MAE_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/MSE_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_10_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/Precision_at_5_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/SAE_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/SSE_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_Adult_n12.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_California_n8.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_Communities_n101.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_Correlated_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_Diabetes_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_IRIS_n4.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_Independent_n60.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_NHANES_n79.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_SOUM_n10.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_SOUM_n6.png
benchmark/results/sv_sweep_20260806_160102/plots/runtime_SOUM_n8.png
benchmark/results/sv_sweep_20260806_160102/results.csv
docs/source/references.bib
examples/approximators/plot_polyshap.py
leverageshap_discussion.pdf
leverageshap_summary.pdf
oddshap_summary.md
polyshap_summary.md
pyproject.toml
src/shapiq/approximator/__init__.py
src/shapiq/approximator/regression/__init__.py
src/shapiq/approximator/regression/base.py
src/shapiq/approximator/regression/leverageshap.py
src/shapiq/approximator/regression/oddshap.py
src/shapiq/approximator/regression/polyshap.py
src/shapiq_benchmark/__init__.py
src/shapiq_benchmark/_optional.py
src/shapiq_benchmark/optimization/optuna_optimization.py
src/shapiq_benchmark/setup.py
src/shapiq_benchmark/tabpfn_bench.py
src/shapiq_games/benchmark/_setup/_vit_setup.py
tests/shapiq/tests_unit/tests_approximators/test_approximator_leverageshap.py
tests/shapiq/tests_unit/tests_approximators/test_approximator_oddshap.py
tests/shapiq/tests_unit/tests_approximators/test_approximator_polyshap.py
tests/shapiq/tests_unit/tests_approximators/test_approximator_regression_base.py
tests/shapiq_benchmark/__init__.py
tests/shapiq_benchmark/test_optional_dependencies.py
tests/shapiq_benchmark/test_setup.py
uv.lock
"""  # <-- PASTE YOUR FULL GIT DIFF LIST HERE BETWEEN TRIPLE QUOTES

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_PDF = "appendix.pdf"
LINES_PER_CODE_SLIDE = 48  # Capacity in 16:9 format

# 16:9 Aspect Ratio Dimensions in Points (960 x 540 pt)
PAGE_WIDTH = 960
PAGE_HEIGHT = 540
PAGE_SIZE = (PAGE_WIDTH, PAGE_HEIGHT)

# Margins
LEFT_MARGIN = 35
RIGHT_MARGIN = 35
TOP_MARGIN = 28
BOTTOM_MARGIN = 28

USABLE_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
USABLE_HEIGHT = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN - 35  # Account for header

# ==============================================================================
# SYNTAX HIGHLIGHTER (Pygments Token Map -> HTML Colors)
# ==============================================================================
TOKEN_COLORS = {
    Token.Keyword: "#d73a49",  # Red (def, return, import, if, class)
    Token.Name.Function: "#6f42c1",  # Purple (function names)
    Token.Name.Class: "#6f42c1",  # Purple (class names)
    Token.String: "#032f62",  # Dark Blue (strings, docstrings)
    Token.Comment: "#6a737d",  # Grey (comments)
    Token.Number: "#005cc5",  # Blue (numbers)
    Token.Operator: "#d73a49",  # Red (=, +, -, etc.)
    Token.Name.Builtin: "#005cc5",  # Blue (len, range, int, str)
    Token.Name.Decorator: "#e36209",  # Orange (@classmethod)
}


def highlight_python_to_html(code_str: str) -> str:
    """Highlight Python code string to ReportLab HTML formatted text."""
    tokens = lex(code_str, PythonLexer())
    html_parts = []

    for token_type, value in tokens:
        # Escape XML special characters and preserve formatting
        escaped = html.escape(value).replace(" ", "&nbsp;").replace("\n", "<br/>")

        # Match Pygments token type to color
        color = None
        for ttype, col in TOKEN_COLORS.items():
            if token_type in ttype:
                color = col
                break

        if color:
            html_parts.append(f'<font color="{color}">{escaped}</font>')
        else:
            html_parts.append(escaped)

    return "".join(html_parts)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def parse_file_list(raw_text: str) -> tuple[list[Path], list[Path]]:
    """Parse git diff output, filter for existing .png and .py files."""
    png_files = []
    py_files = []

    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            png_files.append(p)
        elif p.suffix.lower() == ".py":
            py_files.append(p)

    return png_files, py_files


def calculate_py_chunks(py_path: Path, max_lines: int) -> list[tuple[int, int, str]]:
    """Split python file into chunks of max_lines."""
    if not py_path.exists():
        return [(1, 1, f"# File not found: {py_path}")]

    try:
        content = py_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [(1, 1, f"# Error reading file: {e}")]

    lines = content.splitlines()
    if not lines:
        return [(1, 1, "# Empty file")]

    total_chunks = math.ceil(len(lines) / max_lines)
    chunks = []

    for i in range(total_chunks):
        chunk_lines = lines[i * max_lines : (i + 1) * max_lines]
        chunk_str = "\n".join(chunk_lines)
        chunks.append((i + 1, total_chunks, chunk_str))

    return chunks


# ==============================================================================
# MAIN GENERATOR
# ==============================================================================
def build_appendix_pdf():
    png_files, py_files = parse_file_list(RAW_GIT_DIFF)

    print(
        f"Processing {len(png_files)} PNG files and {len(py_files)} Python files in 16:9 format..."
    )

    # --------------------------------------------------------------------------
    # Pre-calculate Page Ranges for Table of Contents
    # --------------------------------------------------------------------------
    current_page = 2  # Page 1 is TOC
    folder_ranges: dict[str, list[int]] = {}

    # Track PNG pages
    for p in png_files:
        folder = str(p.parent) if str(p.parent) != "." else "Root"
        if folder not in folder_ranges:
            folder_ranges[folder] = [current_page, current_page]
        else:
            folder_ranges[folder][1] = current_page
        current_page += 1

    # Track Py pages
    for p in py_files:
        folder = str(p.parent) if str(p.parent) != "." else "Root"
        chunks = calculate_py_chunks(p, LINES_PER_CODE_SLIDE)
        num_pages = len(chunks)

        start_p = current_page
        end_p = current_page + num_pages - 1

        if folder not in folder_ranges:
            folder_ranges[folder] = [start_p, end_p]
        else:
            folder_ranges[folder][1] = end_p

        current_page += num_pages

    # --------------------------------------------------------------------------
    # Document Setup & Styles
    # --------------------------------------------------------------------------
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=PAGE_SIZE,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=HexColor("#1b5e20"),
    )

    header_style = ParagraphStyle(
        "SlideHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=HexColor("#222222"),
    )

    code_style = ParagraphStyle(
        "HighlightedCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=HexColor("#111111"),
    )

    story = []

    # --------------------------------------------------------------------------
    # SLIDE 1: Table of Contents
    # --------------------------------------------------------------------------
    story.append(Paragraph("[Page 1] Appendix — Table of Contents", title_style))
    story.append(Spacer(1, 8))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=HexColor("#2ca02c"),
            spaceAfter=12,
        )
    )

    toc_data = [["Folder Path", "Slide / Page Range"]]
    for folder, (start_p, end_p) in folder_ranges.items():
        page_str = f"Page {start_p}" if start_p == end_p else f"Pages {start_p} – {end_p}"
        toc_data.append([folder, page_str])

    toc_table = Table(toc_data, colWidths=[USABLE_WIDTH * 0.72, USABLE_WIDTH * 0.28])
    toc_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8f5e9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#1b5e20")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
            ]
        )
    )
    story.append(toc_table)
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PNG SLIDES (1 PNG per 16:9 slide)
    # --------------------------------------------------------------------------
    page_counter = 2

    for p in png_files:
        story.append(Paragraph(f"[{page_counter}] {p.as_posix()}", header_style))
        story.append(Spacer(1, 4))

        if p.exists():
            try:
                with PILImage.open(p) as img:
                    img_w, img_h = img.size

                aspect = img_h / float(img_w)
                target_w = USABLE_WIDTH
                target_h = target_w * aspect

                if target_h > USABLE_HEIGHT:
                    target_h = USABLE_HEIGHT
                    target_w = target_h / aspect

                story.append(Image(p.as_posix(), width=target_w, height=target_h))
            except Exception as e:
                story.append(Paragraph(f"Error rendering image {p}: {e}", styles["Normal"]))
        else:
            story.append(Paragraph(f"File not found on disk: {p.as_posix()}", styles["Normal"]))

        story.append(PageBreak())
        page_counter += 1

    # --------------------------------------------------------------------------
    # PYTHON SLIDES (16:9 Widescreen + Pygments Syntax Highlighting)
    # --------------------------------------------------------------------------
    for p in py_files:
        chunks = calculate_py_chunks(p, LINES_PER_CODE_SLIDE)

        for chunk_idx, total_chunks, chunk_code in chunks:
            part_str = f" (Part {chunk_idx}/{total_chunks})" if total_chunks > 1 else ""
            header_text = f"[{page_counter}] {p.as_posix()}{part_str}"

            story.append(Paragraph(header_text, header_style))
            story.append(Spacer(1, 4))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=HexColor("#cccccc"),
                    spaceAfter=6,
                )
            )

            # Apply Syntax Highlighting
            highlighted_html = highlight_python_to_html(chunk_code)
            story.append(Paragraph(highlighted_html, code_style))

            story.append(PageBreak())
            page_counter += 1

    # Build PDF
    doc.build(story)
    print(f"✓ Success! Generated 16:9 Widescreen PDF '{OUTPUT_PDF}' ({page_counter - 1} pages).")


if __name__ == "__main__":
    build_appendix_pdf()
