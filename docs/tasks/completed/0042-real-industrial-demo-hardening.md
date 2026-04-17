# 0042: Real Industrial Demo Hardening

## Status
Completed

## Why
The first real NEU-backed Demo 02 and Demo 08 paths worked mechanically, but
they still did not make the intended point clearly enough. The broad binary NEU
grouping was weak, the shortcut artefact was too subtle, and the intervention
path had drifted into a training recipe that was harder to present honestly.

## Changes
- Added [balanced_label_subset](/Users/johngoodacre/work/xai/src/xai_demo_suite/data/industrial_manifest.py)
  so the real industrial reports can cap training data without collapsing class
  balance.
- Hardened [neu_cls.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/data/downloaders/neu_cls.py)
  around a curated real shortcut slice:
  - use `scratches` versus `inclusion`,
  - keep the original split-aware archive handling,
  - replace the small corner cue with a full-height correlated border stripe.
- Updated [shortcut_industrial.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/shortcut_industrial.py)
  to:
  - default to the stronger real-data settings,
  - use balanced train capping,
  - label the real path explicitly as the NEU scratches-versus-inclusion slice,
  - report clean, swapped-shortcut, and no-stamp accuracies directly,
  - tighten the presenter-facing wording so the intervention is described as a
    partial fix rather than a benchmark-quality solution.
- Updated [explanation_drift.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/explanation_drift.py)
  so Demo 08 uses the same curated real classifier path and reports clean
  accuracy for the baseline summary rather than blending clean and challenge
  cases together.
- Fixed [augment_stamp_invariant_samples](/Users/johngoodacre/work/xai/src/xai_demo_suite/models/classification/industrial_neural.py)
  so the intervention trains on the original images plus the shortcut-randomised
  variants, not only on the augmented samples.
- Extended [seed_everything](/Users/johngoodacre/work/xai/src/xai_demo_suite/utils/seeds.py)
  to seed PyTorch and request deterministic algorithms, then moved the real
  industrial report defaults to a better stable seed.
- Added and updated focused coverage in
  [test_neu_cls_downloader.py](/Users/johngoodacre/work/xai/tests/unit/test_neu_cls_downloader.py),
  [test_shortcut_industrial_report.py](/Users/johngoodacre/work/xai/tests/unit/test_shortcut_industrial_report.py),
  and
  [test_explanation_drift_report.py](/Users/johngoodacre/work/xai/tests/unit/test_explanation_drift_report.py).

## Output check
- Rebuilt the local NEU manifest at
  `data/processed/neu_cls/shortcut_binary/manifest.jsonl`.
- Regenerated [Demo 02](/Users/johngoodacre/work/xai/outputs/shortcut_industrial/index.html)
  on the curated real slice. The resulting report now makes the intended point:
  clean accuracy is strong for both models, the baseline collapses on
  swapped-shortcut cases, and the intervention only partly recovers that loss.
- Regenerated [Demo 08](/Users/johngoodacre/work/xai/outputs/explanation_drift/index.html)
  so the classifier drift section uses the same real slice and the same clearer
  naming.

## Validation
1. `./.venv/bin/pytest tests/unit/test_neu_cls_downloader.py tests/unit/test_shortcut_industrial_report.py tests/unit/test_explanation_drift_report.py -q`
2. `./.venv/bin/ruff check src tests`
3. `./.venv/bin/mypy src`
4. `./.venv/bin/xai-demo-data prepare neu_cls --category shortcut_binary --overwrite`
5. `./.venv/bin/xai-demo-report shortcut-industrial`
6. `./.venv/bin/xai-demo-report explanation-drift`
7. `./.venv/bin/xai-demo-report verify`
