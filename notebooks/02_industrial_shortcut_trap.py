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
# # Demo 02: Industrial Shortcut Trap
#
# ## Learning goals
# - show how a classifier can learn a spurious production artefact;
# - compare a shortcut-heavy model with a shortcut-resistant intervention;
# - inspect attribution and perturbation evidence around the artefact.
#
# ## Why this demo matters
# It is the industrial analogue of a background shortcut story, but with a
# report shape that is credible for manufacturing data.
#
# ## Dataset and task definition
# The report uses a prepared NEU-CLS-derived binary manifest when available and
# falls back to a synthetic industrial shortcut generator for fresh clones.
#
# ## Model and explanation methods
# Small local convolutional probes with Grad-CAM, Integrated Gradients, and
# targeted masking diagnostics over the stamp and part regions.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.shortcut_industrial import (
    IndustrialShortcutReportConfig,
    build_industrial_shortcut_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))
MISSING_MANIFEST = OUTPUT_ROOT / "_missing_data" / "manifest.jsonl"

config = IndustrialShortcutReportConfig(
    output_dir=OUTPUT_ROOT / "shortcut_industrial",
    synthetic_dir=OUTPUT_ROOT / "shortcut_industrial" / "synthetic",
    input_size=64 if SMOKE_MODE else 128,
    batch_size=4 if SMOKE_MODE else 16,
    epochs=2 if SMOKE_MODE else 18,
    weights_name=None,
    real_manifest_path=MISSING_MANIFEST if SMOKE_MODE else Path(
        "data/processed/neu_cls/shortcut_binary/manifest.jsonl"
    ),
    use_real_data=not SMOKE_MODE,
)

# %% [markdown]
# ## Baseline result
# Build the report and inspect the baseline challenge accuracy first.

# %%
report_path = build_industrial_shortcut_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# Look at swapped-stamp and no-stamp challenge slices rather than only aggregate
# accuracy.
#
# ## Intervention
# Compare the stamp-randomised training path and the masking diagnostics.
#
# ## Re-test
# Rebuild after preparing the NEU-CLS shortcut manifest.
#
# ## What we learned
# Real-image shortcuts become much easier to explain once the report keeps the
# challenge slices explicit.
#
# ## Residual risks and next questions
# The NEU-derived binary task is a local demo slice, not a canonical upstream
# classification benchmark.
