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
# # Demo 08: Explanation Drift Under Shift
#
# ## Learning goals
# - compare prediction drift and explanation drift under nuisance shifts;
# - inspect classifier drift separately from anomaly-detection drift;
# - see how cross-dataset anomaly sections extend the same report contract.
#
# ## Why this demo matters
# Stable-looking predictions can still hide unstable explanations.
#
# ## Dataset and task definition
# The classifier path uses the industrial shortcut report inputs. Optional local
# anomaly sections appear when MVTec AD, MVTec AD 2, or VisA manifests exist.
#
# ## Model and explanation methods
# Learned classifier perturbation summaries plus optional PatchCore-style drift
# diagnostics for anomaly datasets.

# %%
import os
from pathlib import Path

from xai_demo_suite.reports.explanation_drift import (
    ExplanationDriftReportConfig,
    build_explanation_drift_report,
)

SMOKE_MODE = os.environ.get("XAI_DEMO_NOTEBOOK_SMOKE") == "1"
OUTPUT_ROOT = Path(os.environ.get("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", "outputs"))
MISSING_MANIFEST = OUTPUT_ROOT / "_missing_data" / "manifest.jsonl"
MISSING_ROOT = OUTPUT_ROOT / "_missing_data_root"

config = ExplanationDriftReportConfig(
    output_dir=OUTPUT_ROOT / "explanation_drift",
    synthetic_dir=OUTPUT_ROOT / "explanation_drift" / "synthetic",
    classifier_input_size=64 if SMOKE_MODE else 128,
    classifier_batch_size=4 if SMOKE_MODE else 16,
    classifier_epochs=2 if SMOKE_MODE else 18,
    industrial_manifest_path=MISSING_MANIFEST if SMOKE_MODE else Path(
        "data/processed/neu_cls/shortcut_binary/manifest.jsonl"
    ),
    mvtec_manifest_path=MISSING_MANIFEST if SMOKE_MODE else Path(
        "data/processed/mvtec_ad/bottle/manifest.jsonl"
    ),
    mvtec_cache_path=OUTPUT_ROOT / "_artefacts" / "drift_mvtec_bank.npz",
    mvtec_ad_2_processed_root=MISSING_ROOT if SMOKE_MODE else Path("data/processed/mvtec_ad_2"),
    mvtec_ad_2_cache_root=OUTPUT_ROOT / "_artefacts" / "mvtec_ad_2",
    visa_processed_root=MISSING_ROOT if SMOKE_MODE else Path("data/processed/visa"),
    visa_cache_root=OUTPUT_ROOT / "_artefacts" / "visa",
    include_mvtec_if_available=not SMOKE_MODE,
)

# %% [markdown]
# ## Baseline result
# Build the report and compare clean performance against explanation drift.

# %%
report_path = build_explanation_drift_report(config)
print(report_path)

# %% [markdown]
# ## Failure or pitfall
# A stable headline metric can hide unstable evidence.
#
# ## Intervention
# Track perturbation-specific attribution drift and anomaly drift together.
#
# ## Re-test
# Rebuild after preparing local anomaly datasets or changing the perturbation
# mix.
#
# ## What we learned
# Robustness should cover the explanation path as well as the prediction path.
#
# ## Residual risks and next questions
# A local drift report is still not a substitute for a full corruption
# benchmark.
