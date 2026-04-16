# Demo catalogue

Use `./.venv/bin/xai-demo-report suite` to regenerate the synthetic reports and
`./.venv/bin/xai-demo-report verify` to check generated report integrity. See
`docs/DEMO_STATUS.md` for the current runnable status. For the strongest local
presentation suite, use the `suite --include-mvtec` command with the pretrained
MVTec flags listed in `docs/DEMO_STATUS.md`.

## Demo 01 — Waterbirds shortcut
A canonical background-spurious classification story.

Current artefacts:
- static synthetic proxy report: `outputs/waterbirds_shortcut/index.html`
- demo card: `outputs/waterbirds_shortcut/demo_card.html`

The current implementation uses the prepared Waterbirds manifest when present,
with frozen ResNet-18 probes, worst-group metrics, Grad-CAM, Integrated
Gradients, and perturbation checks. When the prepared MetaShift manifest is
also present, the same report adds a natural-context extension on the
cat-vs-dog indoor/outdoor split. Fresh clones still fall back to the synthetic
proxy.

## Demo 02 — Industrial shortcut trap
A classifier learns border, watermark, or fixture leakage.

Current artefacts:
- static report: `outputs/shortcut_industrial/index.html`
- demo card: `outputs/shortcut_industrial/demo_card.html`

The first implementation uses deterministic synthetic part images with a corner
stamp shortcut and a central-shape intervention.

## Demo 03 — PatchCore on MVTec AD
The hero demo: anomaly map plus nearest normal patch provenance. The report can
run deterministic local features for fresh clones or an explicit pretrained
ResNet-18 dense feature-map path for the stronger local anomaly detector.

Current artefacts:
- narrative notebook: `notebooks/03_patchcore_mvtec_bottle.ipynb`
- static report: `outputs/patchcore_bottle/index.html`
- demo card: `outputs/patchcore_bottle/demo_card.html`
- local report index: `outputs/index.html`

The generated `outputs/` files are local artefacts and are ignored by git.

## Demo 04 — PatchCore learns the wrong normal
Nominal-set nuisance contamination and false positives.

Current artefacts:
- static report: `outputs/patchcore_wrong_normal/index.html`
- demo card: `outputs/patchcore_wrong_normal/demo_card.html`

The first implementation uses deterministic slot boards with a corner-tab
nuisance in the nominal set and compares clean versus contaminated memory-bank
provenance.

## Demo 05 — PatchCore cannot count
Repeated-object anomaly layouts expose non-count semantics.

Current artefacts:
- static report: `outputs/patchcore_limits/index.html`
- demo card: `outputs/patchcore_limits/demo_card.html`

The first implementation uses deterministic synthetic slot boards. It also
touches the Demo 06 severity and Demo 07 logic themes by including a scratch
case and a component-identity swap case.

## Demo 06 — PatchCore does not know severity
Novelty score versus severity mismatch.

Current artefacts:
- static synthetic report: `outputs/patchcore_severity/index.html`
- demo card: `outputs/patchcore_severity/demo_card.html`

The first implementation uses a controlled scratch sweep to compare synthetic
severity-area metadata with PatchCore-style patch novelty scores.

## Demo 07 — PatchCore struggles with logical anomalies
MVTec LOCO AD and symbolic / relational failure cases.

Current artefacts:
- static MVTec LOCO report when data is prepared: `outputs/patchcore_logic/index.html`
- demo card: `outputs/patchcore_logic/demo_card.html`

The current local implementation uses MVTec LOCO AD `juice_bottle` structural
and logical anomaly examples when the prepared manifest exists, and now adds a
category-specific front-label template comparator so the report can contrast
PatchCore patch novelty with a narrow packaging-rule check. Fresh clones fall
back to the synthetic slot-board proxy until LOCO data is fetched.

## Demo 08 — Explanation drift under shift
Prediction drift and explanation drift compared under nuisance changes.

Current artefacts:
- static report: `outputs/explanation_drift/index.html`
- demo card: `outputs/explanation_drift/demo_card.html`

The current implementation uses learned industrial shortcut perturbations for
the classifier path, an optional MVTec AD bottle anomaly-drift section, and now
optional second-wave MVTec AD 2 scenario sections when those local manifests
are prepared. Prepared VisA manifests add cross-dataset anomaly-drift sections
to the same report.
