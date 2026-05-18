# Notebook storylines

The notebooks are the primary demo surface for this repository. They are
organised by storyline rather than kept in one flat directory.

## Recommended walkthrough order

0. `xai_demo.ipynb` — presentation-oriented Demo 00 companion notebook, with
   saved story assets under `notebooks/outputs/demo00_story_assets/`.
1. `shortcut_lab/00_moons_stars_clever_hans.ipynb` — no-permission controlled absolute-position shortcut opener, with movement, morph, saliency, representation, and evidence-removal probes.
2. `shortcut_lab/02_industrial_shortcut_trap.ipynb` — real industrial side-band marker shortcut.
3. `shortcut_lab/01_waterbirds_shortcut.ipynb` — natural benchmark shortcut audit.
4. `patchcore_explainability/03_patchcore_mvtec_ad.ipynb` — anomaly maps plus nearest-normal provenance.
5. `patchcore_explainability/04_patchcore_wrong_normal.ipynb` — memory-bank contamination.
6. `patchcore_limits/05_patchcore_count_limit.ipynb` — count limits.
7. `patchcore_limits/06_patchcore_severity_limit.ipynb` — severity limits.
8. `patchcore_limits/07_patchcore_loco_logic_limit.ipynb` — logic and semantic-location limits.
9. `robustness_drift/08_explanation_drift.ipynb` — explanation drift.
10. `data_scouting/90_ieee_dataset_scouting.ipynb` — IEEE dataset candidate register.

The SHAP notebook under `global_local_explainability/` is a focused concept
notebook rather than part of the main demo arc.

## Data-status standard

Each notebook should make its data state visible near the top:

- `DEMO`
- `DATA_MODE`
- `EXTERNAL_DATA_REQUIRED`
- `MANIFEST_PATH`
- `MANIFEST_EXISTS`
- `PROJECT_ROOT`
- `DATASET_SOURCE`
- `LICENCE_NOTE`
- `MISSING_FILES`
- `SEED`

Generated notebooks must declare that no external data are required. Real-data
notebooks should fail clearly if required manifests are missing.

## Notebook contract

For demos `00` to `08`, the notebook itself is expected to carry the runnable
story, visible code, and graphics. These notebooks should not be thin wrappers
around generated HTML reports.
