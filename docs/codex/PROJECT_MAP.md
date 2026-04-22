# Project Map

## Purpose

This repository is a curated XAI demo suite for vision. The active surface on
`main` is a set of storyline notebooks plus the shared data and model tooling
that supports them. The old report-building stack still exists, but it is now
secondary to the notebooks rather than the primary demo interface.

## Top-level layout

- `src/xai_demo_suite/`
  - package code for data adapters, explainability logic, report builders, and
    CLI entry points
- `tests/`
  - unit, integration, and notebook-smoke coverage
- `notebooks/`
  - storyline-based narrative notebooks and the active demo surface
- `docs/`
  - specs, ADRs, task history, review notes, and runbooks
- `data/`
  - raw, processed, synthetic, and artefact storage

## Active notebook directories

- `notebooks/overview/`
  - repo and suite orientation
- `notebooks/shortcut_lab/`
  - Demo 01 Waterbirds shortcut and Demo 02 industrial shortcut trap
- `notebooks/patchcore_explainability/`
  - Demos 03 and 04
- `notebooks/patchcore_limits/`
  - Demos 05 to 07
- `notebooks/robustness_drift/`
  - Demo 08
- `notebooks/global_local_explainability/`
  - SHAP and global-vs-local reference work

## Active entry points

- `src/xai_demo_suite/cli/demo.py`
  - `xai-demo-report` entry point for suite/report generation and verification
- `src/xai_demo_suite/cli/data.py`
  - `xai-demo-data` entry point for dataset listing, fetch, and prepare flows

## Core report modules

- `src/xai_demo_suite/reports/waterbirds_shortcut.py`
  - Demo 01, shortcut classification and interventions
- `src/xai_demo_suite/reports/shortcut_industrial.py`
  - Demo 02, industrial shortcut trap
- `src/xai_demo_suite/reports/patchcore_bottle.py`
  - Demo 03, PatchCore-style anomaly evidence and provenance
- `src/xai_demo_suite/reports/patchcore_wrong_normal.py`
  - Demo 04, nominal-set contamination
- `src/xai_demo_suite/reports/patchcore_limits.py`
  - Demo 05, PatchCore count and logic limits
- `src/xai_demo_suite/reports/patchcore_severity.py`
  - Demo 06, severity mismatch
- `src/xai_demo_suite/reports/patchcore_logic.py`
  - Demo 07, LOCO and logic-aware limits
- `src/xai_demo_suite/reports/explanation_drift.py`
  - Demo 08, prediction and explanation drift
- `src/xai_demo_suite/reports/suite.py`
  - suite index, orchestration, and cross-demo packaging
- `src/xai_demo_suite/reports/review_pack.py`
  - compact external review artefacts
- `src/xai_demo_suite/reports/report_chrome.py`
  - shared presentation frame for flagship reports

## Shared support modules

- `src/xai_demo_suite/data/manifests.py`
  - dataset registry, fetch, and prepare plumbing
- `src/xai_demo_suite/data/waterbirds_manifest.py`
  - Waterbirds-specific manifest logic
- `src/xai_demo_suite/data/industrial_manifest.py`
  - real industrial shortcut manifest logic
- `src/xai_demo_suite/explain/contracts.py`
  - shared explanation-contract structures
- `src/xai_demo_suite/explain/counterfactuals.py`
  - counterfactual helpers
- `src/xai_demo_suite/explain/drift.py`
  - drift metrics and comparisons
- `src/xai_demo_suite/evaluate/localisation.py`
  - localisation checks for anomaly examples
- `src/xai_demo_suite/models/component_rules.py`
  - narrow logic-aware comparator utilities for Demo 07
- `src/xai_demo_suite/vis/image_panels.py`
  - figure assembly and shared visual helpers

## Notebook layer

The notebooks under `notebooks/` are the active narrative and demo layer.
Recent work deliberately made Demo 01 and Demo 02 notebook-native and
self-contained around real Waterbirds and real NEU shortcut data, while the
repo still tries to keep reusable primitives in package code where that does
not undermine the notebook-as-demo goal.

### Demo 01

- real-data-only Waterbirds notebook
- frozen pretrained ResNet-18 features plus logistic-regression heads
- group metrics, occlusion, perturbation, exemplar retrieval, and mitigation

### Demo 02

- real-data-only NEU shortcut notebook
- prepared shortcut manifest under `data/processed/neu_cls/shortcut_binary/`
- controlled coloured side-band marker nuisance on real industrial images
- frozen pretrained ResNet-18 features plus logistic-regression heads
- exact same-image marker swaps, marker ablation, occlusion, exemplar
  retrieval, and mitigation re-test

## Tests most relevant to the active surface

- `tests/integration/test_cli_end_to_end.py`
  - CLI suite, verify, and review-pack smoke path
- `tests/unit/test_notebook_smoke.py`
  - notebook execution smoke
- `tests/unit/test_suite_reports.py`
  - suite index and cross-demo assembly
- `tests/unit/test_patchcore_report.py`
  - core PatchCore report behaviour
- `tests/unit/test_waterbirds_shortcut_report.py`
  - Demo 01 report behaviour
- `tests/unit/test_shortcut_industrial_report.py`
  - Demo 02 and Demo 08 industrial report behaviour
- `tests/unit/test_patchcore_logic_report.py`
  - Demo 07 logic report behaviour
- `tests/unit/test_review_pack.py`
  - review-pack structure

## Local artefacts and generated outputs

Prepared dataset artefacts, feature caches, and notebook outputs are local
runtime products. Checked-in `outputs/` HTML is no longer the main review
surface and should not be treated as the current source of truth.
