# Demo catalogue

Use `./.venv/bin/xai-demo-report suite` to regenerate the synthetic reports and
`./.venv/bin/xai-demo-report verify` to check generated report integrity. See
`docs/DEMO_STATUS.md` for the current runnable status. For the strongest local
presentation suite, use the `suite --include-mvtec` command with the pretrained
MVTec flags listed in `docs/DEMO_STATUS.md`. The flagship reports share a
common presentation chrome with a local-hub link, a concise demo brief, and
related-demo hand-offs for live walkthroughs.

## Demo 01 — Waterbirds shortcut
A canonical background-spurious classification story.

Current artefacts:
- narrative notebook: `notebooks/01_waterbirds_shortcut.ipynb`
- paired percent source: `notebooks/01_waterbirds_shortcut.py`
- static synthetic proxy report: `outputs/waterbirds_shortcut/index.html`
- demo card: `outputs/waterbirds_shortcut/demo_card.html`

The current implementation uses the prepared Waterbirds manifest when present,
with configurable ResNet-18 tuning, worst-group metrics, Grad-CAM, Integrated
Gradients, perturbation checks, and a prototype-exemplar comparator. When the
prepared MetaShift manifest is also present, the same report adds a
natural-context extension on the cat-vs-dog indoor/outdoor split. Fresh clones
still fall back to the synthetic proxy.

## Demo 02 — Industrial shortcut trap
A classifier learns border, watermark, or fixture leakage.

Current artefacts:
- narrative notebook: `notebooks/02_industrial_shortcut_trap.ipynb`
- paired percent source: `notebooks/02_industrial_shortcut_trap.py`
- static report: `outputs/shortcut_industrial/index.html`
- demo card: `outputs/shortcut_industrial/demo_card.html`

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
- narrative notebook: `notebooks/03_patchcore_mvtec_ad.ipynb`
- paired percent source: `notebooks/03_patchcore_mvtec_ad.py`
- static report: `outputs/patchcore_bottle/index.html`
- demo card: `outputs/patchcore_bottle/demo_card.html`
- local report index: `outputs/index.html`

The generated `outputs/` files are local artefacts and are ignored by git by
default. Curated HTML snapshots may still be force-added for public review
releases.

## Demo 04 — PatchCore learns the wrong normal
Nominal-set nuisance contamination and false positives.

Current artefacts:
- narrative notebook: `notebooks/04_patchcore_wrong_normal.ipynb`
- paired percent source: `notebooks/04_patchcore_wrong_normal.py`
- static report: `outputs/patchcore_wrong_normal/index.html`
- demo card: `outputs/patchcore_wrong_normal/demo_card.html`

The first implementation uses deterministic slot boards with a corner-tab
nuisance in the nominal set and compares clean versus contaminated memory-bank
provenance.

## Demo 05 — PatchCore cannot count
Repeated-object anomaly layouts expose non-count semantics.

Current artefacts:
- narrative notebook: `notebooks/05_patchcore_count_limit.ipynb`
- paired percent source: `notebooks/05_patchcore_count_limit.py`
- static report: `outputs/patchcore_limits/index.html`
- demo card: `outputs/patchcore_limits/demo_card.html`

The first implementation uses deterministic synthetic slot boards. It also
touches the Demo 06 severity and Demo 07 logic themes by including a scratch
case and a component-identity swap case.

## Demo 06 — PatchCore does not know severity
Novelty score versus severity mismatch.

Current artefacts:
- narrative notebook: `notebooks/06_patchcore_severity_limit.ipynb`
- paired percent source: `notebooks/06_patchcore_severity_limit.py`
- static synthetic report: `outputs/patchcore_severity/index.html`
- demo card: `outputs/patchcore_severity/demo_card.html`

The first implementation uses a controlled scratch sweep to compare synthetic
severity-area metadata with PatchCore-style patch novelty scores.

## Demo 07 — PatchCore struggles with logical anomalies
MVTec LOCO AD and symbolic / relational failure cases.

Current artefacts:
- narrative notebook: `notebooks/07_patchcore_loco_logic_limit.ipynb`
- paired percent source: `notebooks/07_patchcore_loco_logic_limit.py`
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
- narrative notebook: `notebooks/08_explanation_drift.ipynb`
- paired percent source: `notebooks/08_explanation_drift.py`
- static report: `outputs/explanation_drift/index.html`
- demo card: `outputs/explanation_drift/demo_card.html`

The current implementation uses learned industrial shortcut perturbations for
the classifier path, switching to the same curated NEU scratches-versus-
inclusion shortcut slice when present, an optional MVTec AD bottle
anomaly-drift section, and now optional second-wave MVTec AD 2 scenario
sections when those local manifests are prepared. Prepared VisA manifests add
cross-dataset anomaly-drift sections to the same report.
The classifier path can also be pointed at the prepared KolektorSDD2 shortcut
manifest for a second real industrial source.
