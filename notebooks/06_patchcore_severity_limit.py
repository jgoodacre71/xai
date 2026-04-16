# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Demo 06: PatchCore Severity Limit
#
# ## Learning goals
# - compare severity ranking with novelty ranking;
# - inspect cases where the most severe synthetic defect is not the highest
#   scored anomaly;
# - separate localisation from calibration.
#
# ## Why this demo matters
# It stops users from reading anomaly score as if it were a defect-severity
# scale.
#
# ## Dataset and task definition
# Synthetic severity sweeps with controlled scratch area.
#
# ## Model and explanation methods
# Deterministic PatchCore-style overlays with severity-area metadata.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.patchcore_severity import (
    PatchCoreSeverityReportConfig,
    build_patchcore_severity_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))

config = PatchCoreSeverityReportConfig(
    output_dir=OUTPUT_ROOT / "patchcore_severity",
    synthetic_dir=OUTPUT_ROOT / "patchcore_severity" / "synthetic",
    cache_path=OUTPUT_ROOT / "_artefacts" / "patchcore_severity_bank.npz",
    use_cache=not SMOKE_MODE,
)

# %% [markdown]
# ## Baseline result
# Build the report and compare severity order against score order.

# %%
report_path = build_patchcore_severity_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# Novelty score is not a calibrated severity score.
#
# ## Intervention
# Add an explicit severity model if the product needs that output.
#
# ## Re-test
# Compare the ranking sections after changing the synthetic sweep.
#
# ## What we learned
# Local explanation does not automatically give calibrated intensity.
#
# ## Residual risks and next questions
# Real severity depends on domain cost, not only visible area.
