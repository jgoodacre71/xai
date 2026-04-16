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
# # Demo 01: Waterbirds Shortcut
#
# ## Learning goals
# - compare average and worst-group performance;
# - inspect Grad-CAM and Integrated Gradients on crossed-group cases;
# - compare ERM, group-balanced training, and prototype evidence.
#
# ## Why this demo matters
# It turns a standard spurious-correlation benchmark into a presenter-ready
# report with metrics, exemplar evidence, and perturbation checks.
#
# ## Dataset and task definition
# Waterbirds is the default real-data path. When the prepared MetaShift manifest
# also exists, the report adds the cat-vs-dog indoor/outdoor context slice.
#
# ## Model and explanation methods
# Local ResNet-18 shortcut models with configurable backbone tuning, Grad-CAM,
# Integrated Gradients, group metrics, and prototype exemplars.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.waterbirds_shortcut import (
    WaterbirdsShortcutReportConfig,
    build_waterbirds_shortcut_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))
MISSING_MANIFEST = OUTPUT_ROOT / "_missing_data" / "manifest.jsonl"

config = WaterbirdsShortcutReportConfig(
    output_dir=OUTPUT_ROOT / "waterbirds_shortcut",
    synthetic_dir=OUTPUT_ROOT / "waterbirds_shortcut" / "synthetic",
    manifest_path=Path("data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl"),
    metashift_manifest_path=MISSING_MANIFEST if SMOKE_MODE else Path(
        "data/processed/metashift/subpopulation_shift_cat_dog_indoor_outdoor/manifest.jsonl"
    ),
    use_real_data=not SMOKE_MODE,
    max_train_records=64 if SMOKE_MODE else 800,
    max_test_records=32 if SMOKE_MODE else 400,
    input_size=96 if SMOKE_MODE else 224,
    batch_size=8 if SMOKE_MODE else 16,
    epochs=2 if SMOKE_MODE else 30,
    weights_name=None if SMOKE_MODE else "DEFAULT",
    backbone_tuning="frozen" if SMOKE_MODE else "layer4",
)

# %% [markdown]
# ## Baseline result
# Build the report and inspect ERM versus group-balanced metrics first.

# %%
report_path = build_waterbirds_shortcut_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# Focus on crossed-group examples, not only average accuracy.
#
# ## Intervention
# Compare the group-balanced run and the prototype evidence section.
#
# ## Re-test
# Re-run after changing `backbone_tuning` or after preparing MetaShift.
#
# ## What we learned
# Worst-group behaviour and evidence panels make shortcut reliance visible.
#
# ## Residual risks and next questions
# This is still a local demo-scale training path, not a full benchmark sweep.
