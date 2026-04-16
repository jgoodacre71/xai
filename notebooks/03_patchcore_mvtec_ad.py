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
# # Demo 03: PatchCore on MVTec AD
#
# ## Learning goals
# - inspect local anomaly maps together with nearest-normal provenance;
# - compare lightweight local features with the stronger pretrained path;
# - read the benchmark diagnostics in the same report as the qualitative cases.
#
# ## Why this demo matters
# This is the flagship anomaly explainability demo in the repository.
#
# ## Dataset and task definition
# Prepared MVTec AD bottle data powers the report. The report stays local and
# never commits raw data, masks, or generated memory banks.
#
# ## Model and explanation methods
# PatchCore-style nearest-neighbour patch retrieval, counterfactual patch
# replacement, localisation checks, and dataset-level test-split diagnostics.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    build_patchcore_bottle_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))

config = PatchCoreBottleReportConfig(
    manifest_path=Path("data/processed/mvtec_ad/bottle/manifest.jsonl"),
    output_dir=OUTPUT_ROOT / "patchcore_bottle",
    cache_path=OUTPUT_ROOT / "_artefacts" / "patchcore_bottle_bank.npz",
    feature_extractor_name="mean_rgb" if SMOKE_MODE else "feature_map_resnet18_pretrained",
    max_train=4 if SMOKE_MODE else 20,
    max_examples=1 if SMOKE_MODE else 3,
    input_size=96 if SMOKE_MODE else 224,
    batch_size=4 if SMOKE_MODE else 8,
    coreset_size=None if SMOKE_MODE else 512,
    max_benchmark_records=4 if SMOKE_MODE else None,
    use_cache=not SMOKE_MODE,
)

# %% [markdown]
# ## Baseline result
# Build the report and read the first anomaly example before the benchmark panel.

# %%
report_path = build_patchcore_bottle_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# Patch novelty is useful evidence, but it is not the same as semantics, logic,
# or calibrated severity.
#
# ## Intervention
# Use the provenance, counterfactual, and benchmark sections together.
#
# ## Re-test
# Switch between the lightweight and pretrained feature extractors.
#
# ## What we learned
# Provenance retrieval makes anomaly explanations much more concrete.
#
# ## Residual risks and next questions
# This remains a local report path rather than an official benchmark
# reproduction.
