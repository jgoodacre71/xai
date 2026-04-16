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
# # XAI Demo Suite Overview
#
# ## Learning goals
# - understand the four flagship pillars in the repo;
# - know which datasets and reports back each pillar;
# - know where to start for a short or long walkthrough.
#
# ## Why this demo matters
# The repository is a curated explainability product rather than a notebook zoo.
# Package code lives in `src/`; notebooks stay thin and narrative-led.
#
# ## Why this demo set matters
# ## Dataset and task definition
# - Demo 01: Waterbirds and optional MetaShift for shortcut learning
# - Demo 02: industrial shortcut trap with synthetic or prepared NEU-CLS slices
# - Demo 03: PatchCore on MVTec AD bottle
# - Demo 07: PatchCore logic limits on MVTec LOCO AD
# - Demo 08: explanation drift across classifier and anomaly paths
#
# ## Model and explanation methods
# - ERM versus group-balanced classifiers with Grad-CAM and Integrated Gradients
# - PatchCore with provenance retrieval and counterfactual patch replacement
# - drift checks under perturbation and dataset shift
#
# ## Baseline result
# Use the local demo hub to inspect the current generated outputs.
#
# ## Failure or pitfall
# Some reports need prepared local datasets; fresh clones fall back to synthetic
# paths where the demo contract allows it.
#
# ## Intervention
# Prepare the local datasets you want to show and regenerate the suite.
#
# ## Re-test
# Run `./.venv/bin/xai-demo-report suite` and `./.venv/bin/xai-demo-report verify`.
#
# ## What we learned
# The codebase is organised so the same report builders power the CLI, tests,
# and notebooks.
#
# ## Residual risks and next questions
# Prepared-data notebooks are local only; raw data and generated outputs remain
# out of git.

# %% [markdown]
# ## Suggested walkthrough order
#
# 1. Demo 03 for the main PatchCore story
# 2. Demo 01 for classical shortcut learning
# 3. Demo 07 for logic limits
# 4. Demo 08 for robustness and explanation drift
