# XAI Demo Suite

This repository contains a curated suite of vision XAI demos focused on
model-behaviour observability:

> What did the model actually learn, what evidence did it use, what would change
> the decision, and what risk remains after mitigation?

The project is not a generic heatmap gallery. Each demo should show an apparent
success, reveal a shortcut or limitation, interrogate the behaviour, apply an
intervention, and re-test what improved and what did not.

## Getting started

Preferred `uv` workflow:

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src
uv run pytest -q
```

For notebook execution and deep-feature demos:

```bash
uv sync --group dev --group notebooks --group ml
```

Plain `venv` fallback:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev,notebooks,ml]"
./.venv/bin/pytest -q
```

## Conceptual demo order

Open the notebooks directly. The conceptual walkthrough order is:

1. `notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb` — no-permission controlled Clever-Hans opener: absolute-position shortcut, movement counterfactuals, response maps, morphs, saliency caveats, representation neighbours/probes, and evidence-removal re-test.
2. `notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb` — real NEU industrial side-band marker shortcut.
3. `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb` — literature-aligned natural Waterbirds shortcut audit.
4. `notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb` — PatchCore anomaly map plus nearest-normal provenance.
5. `notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb` — memory bank learns the wrong normal.
6. `notebooks/patchcore_limits/05_patchcore_count_limit.ipynb` — PatchCore cannot natively count.
7. `notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb` — novelty score is not severity.
8. `notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb` — PatchCore struggles with logical product rules.
9. `notebooks/robustness_drift/08_explanation_drift.ipynb` — predictions can remain stable while evidence drifts.
10. `notebooks/data_scouting/90_ieee_dataset_scouting.ipynb` — IEEE DataPort candidate register, not a modelling demo.

`notebooks/global_local_explainability/09_global_vs_local_explainability_shap.ipynb`
is an additional concept notebook outside the main numbered demo arc.

## Data

Raw datasets are not committed. Start with:

- `docs/DATA_REQUIREMENTS.md`
- `docs/DATA_PERMISSION_MATRIX.md`
- `docs/DATA_REPLICATION_WORKFLOW.md`
- `data_registry.yaml`

Audit local data state with:

```bash
uv run python scripts/audit_data_inventory.py --root .
```

Generated demos, such as Demo 00, require no external data. Real-data notebooks
must fail clearly when their required manifests are missing.

Important permission defaults:

- Waterbirds: verify upstream CUB and Places-derived terms before work use.
- NEU-CLS: verify the official Northeastern terms before workplace redistribution or external publication.
- MVTec AD, MVTec LOCO AD, MVTec AD 2, and KolektorSDD2: record non-commercial restrictions and seek work approval.
- VisA: a more permissive anomaly option to verify internally.
- IEEE DataPort: use as a controlled candidate register; standard datasets may require subscriber access and all datasets require attribution/citation.

## Notebook data status

Every active notebook should show:

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

Generated demos should declare `DATA_MODE: generated_controlled_demo` and
`EXTERNAL_DATA_REQUIRED: false`.

## Legacy HTML reports

The active demo surface is the checked-in notebook set under `notebooks/`.
The `xai-demo-report` commands and modules under `src/xai_demo_suite/reports/`
remain available as a secondary legacy surface, but they are no longer the
primary way to present the project.

## Repository layout

- `notebooks/` — active notebook demo surface.
- `src/xai_demo_suite/` — reusable package code and legacy report builders.
- `tests/` — unit, integration, and notebook smoke tests.
- `docs/` — architecture, XAI contract, data requirements, runbooks, and task memory.
- `data/` — local raw/interim/processed/generated artefact roots; raw datasets are ignored.
