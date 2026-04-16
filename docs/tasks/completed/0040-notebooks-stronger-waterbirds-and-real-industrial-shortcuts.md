# 0040: Notebooks, Stronger Waterbirds, and Real Industrial Shortcuts

## Status
Completed

## Why
The repo now has strong HTML reports, but it still under-delivers on three
important spec promises: a full demo notebook layer, a stronger non-frozen
Waterbirds path, and a real industrial dataset-backed shortcut demo.

## Source docs
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/DATASETS.md
- docs/DEMO_STATUS.md
- docs/DEMO_CATALOGUE.md

## Scope
- Add the missing notebook suite scaffolds so every flagship demo has a thin
  narrative entry point.
- Add a stronger Waterbirds training mode beyond the frozen linear probe path.
- Add local NEU-CLS dataset support and integrate a real-image shortcut path
  into Demo 02, with reuse in Demo 08 where feasible.

## Out of scope
- Remote hosting or deployment.
- Full benchmark reproduction for any external dataset.
- Large notebook outputs checked into git.

## Deliverables
- New notebook files under `notebooks/`
- Updated notebook tests
- Stronger Waterbirds model path in `src/xai_demo_suite/models/classification/`
- NEU-CLS downloader / preparer and manifest loader
- Updated Demo 02 and Demo 08 report logic
- Updated docs and completed task memory

## Constraints
- Keep notebooks thin; package code remains the product.
- Do not commit raw datasets or extracted copies.
- Record research-only or uncertain licence status conservatively.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_notebooks.py tests/unit/test_waterbirds_shortcut_report.py tests/unit/test_shortcut_industrial_report.py tests/unit/test_explanation_drift_report.py tests/unit/test_neu_cls_downloader.py -q`
4. targeted demo builds for Waterbirds, Demo 02, and Demo 08
5. `./.venv/bin/xai-demo-report verify`

## Risks
- Real-image shortcut generation can drift into data-prep complexity if the
  prepared representation is not kept simple.
- Stronger Waterbirds training can materially increase runtime if the tuning
  mode is too aggressive for local CPU runs.

## Progress log
### 2026-04-16
- Started: Implementation pass for notebook coverage, stronger Waterbirds
  training, and a real industrial shortcut dataset path.
- Completed:
  - added NEU-CLS fetch and prepare support plus a real-image binary shortcut
    manifest path for Demo 02 and Demo 08;
  - generalised the industrial classifier label handling beyond `normal` /
    `defect`;
  - added configurable ResNet-18 tuning modes for the real Waterbirds path and
    updated the report copy to match;
  - added the full spec notebook set under `notebooks/` as output-free `.ipynb`
    files plus paired percent-script sources;
  - added notebook structure checks and paired notebook smoke execution tests;
  - updated README, dataset docs, demo status, notebook policy, and dataset
    registry memory.
- Validation:
  1. `./.venv/bin/ruff check src tests notebooks`
  2. `./.venv/bin/mypy src`
  3. `./.venv/bin/pytest tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py tests/unit/test_waterbirds_shortcut_report.py tests/unit/test_shortcut_industrial_report.py tests/unit/test_explanation_drift_report.py tests/unit/test_neu_cls_downloader.py tests/unit/test_suite_reports.py tests/unit/test_synthetic_fixtures.py -q`
  4. `./.venv/bin/xai-demo-report waterbirds-shortcut --no-real-data --weights none --backbone-tuning frozen --epochs 2 --max-train 64 --max-test 32 --input-size 96 --batch-size 8`
  5. `./.venv/bin/xai-demo-report shortcut-industrial --no-real-data --weights none --epochs 2 --input-size 64 --batch-size 4`
  6. `./.venv/bin/xai-demo-report explanation-drift --industrial-manifest-path outputs/_missing_data/manifest.jsonl --mvtec-manifest-path outputs/_missing_data/manifest.jsonl --mvtec-ad2-processed-root outputs/_missing_data_root --visa-processed-root outputs/_missing_data_root`
  7. `./.venv/bin/xai-demo-report verify`
