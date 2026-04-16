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
# # Demo 04: PatchCore Learns the Wrong Normal
#
# ## Learning goals
# - compare clean and contaminated normal sets;
# - inspect how nuisance contamination changes the nearest-normal explanation;
# - separate a useful detector from a bad normal-set definition.
#
# ## Why this demo matters
# It explains a common anomaly-detection failure mode without needing real-data
# contamination in the main branch.
#
# ## Dataset and task definition
# Synthetic nuisance boards with a corner-tab acquisition artefact in part of
# the nominal set.
#
# ## Model and explanation methods
# Deterministic PatchCore-style novelty with provenance overlays and nearest
# normal patch crops.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.patchcore_wrong_normal import (
    PatchCoreWrongNormalReportConfig,
    build_patchcore_wrong_normal_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))

config = PatchCoreWrongNormalReportConfig(
    output_dir=OUTPUT_ROOT / "patchcore_wrong_normal",
    synthetic_dir=OUTPUT_ROOT / "patchcore_wrong_normal" / "synthetic",
    clean_cache_path=OUTPUT_ROOT / "_artefacts" / "wrong_normal_clean.npz",
    contaminated_cache_path=OUTPUT_ROOT / "_artefacts" / "wrong_normal_contaminated.npz",
    use_cache=not SMOKE_MODE,
)

# %% [markdown]
# ## Baseline result
# Build the report and compare the clean-bank and contaminated-bank examples.

# %%
report_path = build_patchcore_wrong_normal_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# A good detector can still fail if the normal-set governance is wrong.
#
# ## Intervention
# Remove nuisance-heavy normals and rebuild the memory bank.
#
# ## Re-test
# Re-run the report after changing the contamination assumptions.
#
# ## What we learned
# Provenance makes normal-set mistakes inspectable rather than abstract.
#
# ## Residual risks and next questions
# Real acquisition nuisances are usually messier than this synthetic slice.
