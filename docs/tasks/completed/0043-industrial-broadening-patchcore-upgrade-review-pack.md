# 0043: Industrial Broadening, PatchCore Upgrade, and Review Pack

## Status
Completed

## Why
The repo had reached a coherent first version, but the next practical quality
gains were clear. The industrial pillar still relied on one curated real-data
path, the PatchCore hero still leaned too heavily on a single category and a
lighter backbone, and there was no compact external-review layer for GitHub or
ChatGPT-based inspection.

## Changes
- Added a second real industrial dataset adapter in
  [ksdd2.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/data/downloaders/ksdd2.py)
  and wired it into [data.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/cli/data.py).
  The adapter prepares a shared shortcut-style manifest at
  `data/processed/ksdd2/shortcut_binary/manifest.jsonl` so Demo 02 and Demo 08
  can point at a second real industrial source without bespoke report logic.
- Extended [patchcore_bottle.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/patchcore_bottle.py)
  and [demo.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/cli/demo.py)
  with stronger feature extractors:
  - `feature_map_wide_resnet50_2_random`
  - `feature_map_wide_resnet50_2_pretrained`
- Made the PatchCore report category-aware so the same report path now renders
  correct titles and summaries for manifests beyond bottle, including capsule.
- Added [review_pack.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/review_pack.py)
  and a new `xai-demo-report review-pack` CLI path that builds a compact local
  review hub for external readers.
- Updated
  [README.md](/Users/johngoodacre/work/xai/README.md),
  [docs/DATASETS.md](/Users/johngoodacre/work/xai/docs/DATASETS.md),
  [docs/DEMO_CATALOGUE.md](/Users/johngoodacre/work/xai/docs/DEMO_CATALOGUE.md),
  [docs/DEMO_STATUS.md](/Users/johngoodacre/work/xai/docs/DEMO_STATUS.md),
  [docs/PATCHCORE_NOTES.md](/Users/johngoodacre/work/xai/docs/PATCHCORE_NOTES.md),
  and [data_registry.yaml](/Users/johngoodacre/work/xai/data_registry.yaml)
  to record the new dataset path, stronger PatchCore options, broader category
  coverage, and the review-pack output.
- Added focused regression coverage in
  [test_ksdd2_downloader.py](/Users/johngoodacre/work/xai/tests/unit/test_ksdd2_downloader.py),
  [test_patchcore_report.py](/Users/johngoodacre/work/xai/tests/unit/test_patchcore_report.py),
  and [test_review_pack.py](/Users/johngoodacre/work/xai/tests/unit/test_review_pack.py).

## Output check
- Prepared an additional local MVTec AD category manifest at
  `data/processed/mvtec_ad/capsule/manifest.jsonl`.
- Regenerated a second visible PatchCore category report at
  [outputs/patchcore_capsule/index.html](/Users/johngoodacre/work/xai/outputs/patchcore_capsule/index.html).
- Regenerated a stronger WideResNet50-2 PatchCore output at
  [outputs/patchcore_bottle_wrn50/index.html](/Users/johngoodacre/work/xai/outputs/patchcore_bottle_wrn50/index.html).
- Added a compact external-review layer at
  [outputs/review_pack/index.html](/Users/johngoodacre/work/xai/outputs/review_pack/index.html).

## Validation
1. `./.venv/bin/pytest tests/unit/test_ksdd2_downloader.py tests/unit/test_patchcore_report.py tests/unit/test_suite_reports.py tests/unit/test_review_pack.py -q`
2. `./.venv/bin/ruff check src tests`
3. `./.venv/bin/mypy src`
4. `./.venv/bin/xai-demo-data prepare mvtec_ad --category capsule --overwrite`
5. `./.venv/bin/xai-demo-report patchcore-bottle --manifest-path data/processed/mvtec_ad/capsule/manifest.jsonl --output-dir outputs/patchcore_capsule --cache-path data/artefacts/patchcore/capsule/report_bank.npz --feature-extractor feature_map_resnet18_pretrained --max-train 20 --max-examples 3 --coreset-size 512 --input-size 224 --no-cache`
6. `./.venv/bin/xai-demo-report patchcore-bottle --manifest-path data/processed/mvtec_ad/bottle/manifest.jsonl --output-dir outputs/patchcore_bottle_wrn50 --cache-path data/artefacts/patchcore/bottle/report_wrn50_bank.npz --feature-extractor feature_map_wide_resnet50_2_pretrained --max-train 10 --max-examples 2 --coreset-size 512 --input-size 224 --no-cache`
7. `./.venv/bin/xai-demo-report review-pack`
8. `./.venv/bin/xai-demo-report verify`
