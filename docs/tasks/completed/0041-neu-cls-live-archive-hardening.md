# 0041: NEU-CLS Live Archive Hardening

## Status
Completed

## Why
The first live NEU-CLS archive fetch exposed two real compatibility gaps:
- the public archive arrived as a ZIP file without a `.zip` suffix;
- the extracted layout used `train/train/images` and `valid/valid/images`
  folders with full class names such as `crazing_10.jpg`, not only the original
  `IMAGES/Cr_000.bmp` layout.

Those gaps blocked local preparation and left Demo 08's report copy out of sync
with the real data path.

## Changes
- Hardened [neu_cls.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/data/downloaders/neu_cls.py)
  so extraction detects ZIP archives by content, not only by suffix.
- Extended the NEU class parser to accept both original short codes and full
  class-name prefixes from split-layout archives.
- Reused explicit `train` / `valid` split folders when present instead of
  forcing a synthetic 70/30 split.
- Updated [explanation_drift.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/explanation_drift.py)
  so the report copy and caveats reflect the real NEU-backed classifier path.
- Added regression coverage in
  [test_neu_cls_downloader.py](/Users/johngoodacre/work/xai/tests/unit/test_neu_cls_downloader.py)
  and
  [test_explanation_drift_report.py](/Users/johngoodacre/work/xai/tests/unit/test_explanation_drift_report.py).

## Live data outcome
- Downloaded a public `NEU-CLS.zip` bundle through the Figshare API-backed file
  URL.
- Prepared `data/processed/neu_cls/shortcut_binary/manifest.jsonl`.
- Generated the real-image Demo 02 report at
  [outputs/shortcut_industrial/index.html](/Users/johngoodacre/work/xai/outputs/shortcut_industrial/index.html).
- Regenerated Demo 08 with the real NEU-backed classifier path at
  [outputs/explanation_drift/index.html](/Users/johngoodacre/work/xai/outputs/explanation_drift/index.html).

## Validation
1. `./.venv/bin/pytest tests/unit/test_neu_cls_downloader.py -q`
2. `./.venv/bin/pytest tests/unit/test_explanation_drift_report.py -k real_industrial_manifest -q`
3. `./.venv/bin/ruff check src/xai_demo_suite/data/downloaders/neu_cls.py src/xai_demo_suite/reports/explanation_drift.py tests/unit/test_neu_cls_downloader.py tests/unit/test_explanation_drift_report.py`
4. `./.venv/bin/xai-demo-data fetch neu_cls --category shortcut_binary --archive-url https://ndownloader.figshare.com/files/54094775 --overwrite`
5. `./.venv/bin/xai-demo-data prepare neu_cls --category shortcut_binary --overwrite`
6. `./.venv/bin/xai-demo-report shortcut-industrial --real-manifest-path data/processed/neu_cls/shortcut_binary/manifest.jsonl --epochs 2 --input-size 96 --batch-size 16 --max-train 400`
7. one-epoch local Python report build for Demo 08 against the prepared NEU manifest
8. `./.venv/bin/xai-demo-report verify`
