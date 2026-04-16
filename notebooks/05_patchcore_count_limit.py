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
# # Demo 05: PatchCore Count Limit
#
# ## Learning goals
# - see why local novelty is not the same as counting;
# - compare count errors with the patches that receive the highest score;
# - prepare the ground for logic and severity limitations.
#
# ## Why this demo matters
# It prevents overclaiming what nearest-neighbour patch novelty can represent.
#
# ## Dataset and task definition
# Synthetic slot boards with missing, extra, scratched, and swapped-component
# cases.
#
# ## Model and explanation methods
# Deterministic PatchCore-style overlays, patch crops, and simple mask overlap
# checks.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.patchcore_limits import (
    PatchCoreLimitsReportConfig,
    build_patchcore_limits_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))

config = PatchCoreLimitsReportConfig(
    output_dir=OUTPUT_ROOT / "patchcore_limits",
    synthetic_dir=OUTPUT_ROOT / "patchcore_limits" / "synthetic",
    cache_path=OUTPUT_ROOT / "_artefacts" / "patchcore_limits_bank.npz",
    use_cache=not SMOKE_MODE,
)

# %% [markdown]
# ## Baseline result
# Build the report and compare count cases before the severity and logic rows.

# %%
report_path = build_patchcore_limits_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# The top patch may be locally novel without answering the count question.
#
# ## Intervention
# Add explicit counting or structure-aware logic on top of the detector.
#
# ## Re-test
# Compare this report with Demo 06 and Demo 07.
#
# ## What we learned
# PatchCore is a useful local novelty tool, not a count engine.
#
# ## Residual risks and next questions
# Real count failures can involve clutter, occlusion, and perspective changes.
