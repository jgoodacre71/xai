# Demo catalogue

The active demo surface is now the notebook tree under `notebooks/`. The
non-SHAP demos are intended to be self-contained notebook walkthroughs with the
story, code, and graphics visible in one place.

The older HTML report stack still exists as a legacy secondary surface, but it
is no longer the primary way to explore the project.

## Demo 01 — Waterbirds shortcut
A canonical background-spurious classification story.

Current artefacts:
- narrative notebook: `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb`

The current implementation uses the prepared Waterbirds manifest when present,
with configurable ResNet-18 tuning, worst-group metrics, Grad-CAM, Integrated
Gradients, perturbation checks, and a prototype-exemplar comparator. When the
prepared MetaShift manifest is also present, the same report adds a
natural-context extension on the cat-vs-dog indoor/outdoor split. Fresh clones
still fall back to the synthetic proxy.

## Demo 02 — Industrial shortcut trap
A classifier learns border, watermark, or fixture leakage.

Current artefacts:
- narrative notebook: `notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb`

The current implementation uses a curated NEU scratches-versus-inclusion
shortcut slice when present, with synthetic fallback for fresh clones. The real
path uses a stronger correlated border stripe, balanced train capping, clean
versus challenge metrics, a learned shortcut model, a shortcut-randomised
intervention model, and known-region diagnostics.
The same report contract now also accepts a prepared KolektorSDD2 shortcut
manifest for a second real industrial source.

## Demo 03 — PatchCore on MVTec AD
The hero demo: anomaly map plus nearest normal patch provenance. The report can
run deterministic local features for fresh clones or an explicit pretrained
ResNet-18 dense feature-map path for the stronger local anomaly detector.

Current artefacts:
- narrative notebook: `notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb`

## Demo 04 — PatchCore learns the wrong normal
Nominal-set nuisance contamination and false positives.

Current artefacts:
- narrative notebook: `notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb`

The first implementation uses deterministic slot boards with a corner-tab
nuisance in the nominal set and compares clean versus contaminated memory-bank
provenance.

## Demo 05 — PatchCore cannot count
Repeated-object anomaly layouts expose non-count semantics.

Current artefacts:
- narrative notebook: `notebooks/patchcore_limits/05_patchcore_count_limit.ipynb`

The first implementation uses deterministic synthetic slot boards. It also
touches the Demo 06 severity and Demo 07 logic themes by including a scratch
case and a component-identity swap case.

## Demo 06 — PatchCore does not know severity
Novelty score versus severity mismatch.

Current artefacts:
- narrative notebook: `notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb`

The first implementation uses a controlled scratch sweep to compare synthetic
severity-area metadata with PatchCore-style patch novelty scores.

## Demo 07 — PatchCore struggles with logical anomalies
MVTec LOCO AD and symbolic / relational failure cases.

Current artefacts:
- narrative notebook: `notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb`

The current local implementation uses MVTec LOCO AD `juice_bottle` structural
and logical anomaly examples when the prepared manifest exists, and now adds a
category-specific front-label template comparator so the report can contrast
PatchCore patch novelty with a narrow packaging-rule check. Fresh clones fall
back to the synthetic slot-board proxy until LOCO data is fetched.

## Demo 08 — Explanation drift under shift
Prediction drift and explanation drift compared under nuisance changes.

Current artefacts:
- narrative notebook: `notebooks/robustness_drift/08_explanation_drift.ipynb`

The current implementation uses learned industrial shortcut perturbations for
the classifier path, switching to the same curated NEU scratches-versus-
inclusion shortcut slice when present, an optional MVTec AD bottle
anomaly-drift section, and now optional second-wave MVTec AD 2 scenario
sections when those local manifests are prepared. Prepared VisA manifests add
cross-dataset anomaly-drift sections to the same report.
The classifier path can also be pointed at the prepared KolektorSDD2 shortcut
manifest for a second real industrial source.

## Additional concept notebook — Global versus local explainability with SHAP

Current artefacts:
- narrative notebook: `notebooks/global_local_explainability/09_global_vs_local_explainability_shap.ipynb`

This notebook is a cleaned storyline notebook rather than part of the main
eight-demo suite. It focuses on a synthetic X-ray setting where the target
shifts from broad global shell evidence to a small local star-like structure,
using SHAP to compare global and local explainability behaviour across linear
models and a CNN.
