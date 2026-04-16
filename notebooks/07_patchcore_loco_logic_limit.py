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
# # Demo 07: PatchCore LOCO Logic Limit
#
# ## Learning goals
# - compare local novelty with logic-aware category rules;
# - inspect structural and logical anomalies on the same report path;
# - see where provenance is useful and where logic modelling is still missing.
#
# ## Why this demo matters
# It is the clearest answer to the question “what PatchCore cannot do on its
# own?”.
#
# ## Dataset and task definition
# The report uses MVTec LOCO AD `juice_bottle` when the prepared manifest is
# available, with a synthetic fallback for fresh clones.
#
# ## Model and explanation methods
# PatchCore-style patch novelty plus a narrow front-label component rule for the
# aligned packaging category.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.patchcore_logic import (
    PatchCoreLogicReportConfig,
    build_patchcore_logic_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))
MISSING_MANIFEST = OUTPUT_ROOT / "_missing_data" / "manifest.jsonl"

config = PatchCoreLogicReportConfig(
    output_dir=OUTPUT_ROOT / "patchcore_logic",
    synthetic_dir=OUTPUT_ROOT / "patchcore_logic" / "synthetic",
    cache_path=OUTPUT_ROOT / "_artefacts" / "patchcore_logic_bank.npz",
    loco_cache_path=OUTPUT_ROOT / "_artefacts" / "patchcore_loco_logic_bank.npz",
    manifest_path=MISSING_MANIFEST if SMOKE_MODE else Path(
        "data/processed/mvtec_loco_ad/juice_bottle/manifest.jsonl"
    ),
    use_cache=not SMOKE_MODE,
)

# %% [markdown]
# ## Baseline result
# Build the report and compare the PatchCore section with the component-rule
# benchmark.

# %%
report_path = build_patchcore_logic_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# High local novelty does not mean the model understands packaging logic.
#
# ## Intervention
# Add explicit component or rule modelling where the task depends on it.
#
# ## Re-test
# Rebuild after preparing LOCO data or changing the rule threshold assumptions.
#
# ## What we learned
# Provenance-rich anomaly maps and logic-aware comparators answer different
# questions.
#
# ## Residual risks and next questions
# The current comparator is intentionally narrow and category-specific.
