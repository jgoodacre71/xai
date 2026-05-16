# Demo catalogue

The active demo surface is the notebook tree under `notebooks/`. The older HTML
report stack remains as a secondary legacy surface.

The conceptual order is controlled shortcut → industrial shortcut → natural
benchmark → anomaly provenance → wrong normal → limits → drift → dataset
scouting.

## Demo 00 — Moons/Stars Clever-Hans two-act laboratory

No-permission generated opener.

Current artefacts:
- notebook: `notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb`

Role:
- show the smallest version of “the model solved the dataset, not the task”;
- Act I uses generated moons/stars with an absolute-position shortcut: moons
  near the lower-left and stars near the upper-right;
- Act II then fools the reader again with a near-invisible background/acquisition
  cue that a CNN can exploit with decisive confidence;
- demonstrate group audit, exact movement counterfactuals, position-response
  maps, shape morphing, saliency caveats, minimal evidence removal,
  representation neighbours/probes, mitigation, and same-case re-test;
- make the governing lesson explicit: the shortcut changes, but the XAI
  discipline stays the same.

## Demo 02 — Industrial side-band marker shortcut

The most visually immediate applied shortcut demo.

Current artefacts:
- notebook: `notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb`

Role:
- use real NEU images with a controlled side-band marker;
- show clean versus challenge metrics, marker counterfactuals, occlusion,
  neighbours, mitigation, and same-case re-test;
- remain a real-image demo, not a synthetic panel demo.

## Demo 01 — Waterbirds shortcut audit

The literature-aligned natural benchmark.

Current artefacts:
- notebook: `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb`

Role:
- use real Waterbirds data only;
- train frozen pretrained ResNet-18 features plus logistic-regression heads;
- show ordinary validation inflation, balanced group audit, selected crossed
  failure, background/context perturbation, exemplar retrieval, advanced model
  interrogation, mitigation, and final self-check.

## Demo 03 — PatchCore anomaly provenance

PatchCore centrepiece: anomaly map plus nearest-normal provenance.

Current artefacts:
- notebook: `notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb`

Data gate:
- use MVTec AD if permitted;
- if MVTec is unavailable or not permitted at work, use VisA or approved IEEE
  anomaly data.

## Demo 04 — PatchCore learns the wrong normal

Memory-bank methods can learn the wrong normal when nominal data are
contaminated.

Current artefacts:
- notebook: `notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb`

## Demo 05 — PatchCore cannot count

Repeated-object layouts expose that PatchCore novelty does not natively count
instances.

Current artefacts:
- notebook: `notebooks/patchcore_limits/05_patchcore_count_limit.ipynb`

## Demo 06 — PatchCore does not know severity

Feature-space novelty is not calibrated engineering severity.

Current artefacts:
- notebook: `notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb`

## Demo 07 — PatchCore struggles with logical anomalies

PatchCore localises unusual regions but does not natively understand symbolic
product rules.

Current artefacts:
- notebook: `notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb`

Data gate:
- use MVTec LOCO AD if permitted;
- if LOCO is unavailable or not permitted at work, use VisA or approved IEEE
  anomaly/inspection data.

## Demo 08 — Explanation drift under shift

Prediction stability and explanation stability are different signals.

Current artefacts:
- notebook: `notebooks/robustness_drift/08_explanation_drift.ipynb`

## Demo 90 — IEEE dataset scouting

Controlled candidate register, not a finished modelling demo.

Current artefacts:
- notebook: `notebooks/data_scouting/90_ieee_dataset_scouting.ipynb`
- register: `data/ieee_candidates.yaml`
- guidance: `docs/IEEE_DATA_SCOUTING.md`

Role:
- record access type, licence, citation, work-permission status, demo fit, and
  next action before any IEEE DataPort dataset is adopted.

## Additional concept notebook — Global versus local explainability with SHAP

Current artefacts:
- notebook: `notebooks/global_local_explainability/09_global_vs_local_explainability_shap.ipynb`

This notebook is outside the main demo arc.
