# 0004-patchcore-feature-extractor-interface: PatchCore feature extractor interface

## Status
Complete

## Owner
Codex thread

## Why
The current PatchCore foundation uses mean RGB patch features so provenance can
be tested cheaply. The next step is to separate patch extraction from feature
extraction and add a deep-feature-ready interface without breaking memory-bank
metadata, nearest-normal retrieval, or tests.

## Source docs
- REPO_SPEC.md
- docs/ARCHITECTURE.md
- docs/PATCHCORE_NOTES.md
- docs/XAI_CONTRACT.md
- docs/tasks/completed/0003-patchcore-provenance-foundation.md

## Scope
- Add a typed patch feature extractor protocol.
- Move mean RGB feature extraction behind that protocol.
- Add an optional Torch/Torchvision feature extractor wrapper that is imported
  lazily, so the base test suite does not require large ML dependencies.
- Refactor memory-bank construction and scoring to accept any extractor.
- Preserve current `build_mean_colour_memory_bank` and
  `score_image_against_memory_bank` convenience functions.
- Add tests proving different extractors share the same provenance path.

## Out of scope
- Installing Torch in this turn.
- Downloading pretrained weights.
- Coreset selection.
- Full anomaly-map rendering.
- Notebook work.

## Deliverables
- `src/xai_demo_suite/models/patchcore/features.py`
- refactored `baseline.py`
- tests for extractor protocol and provenance preservation
- docs update explaining the feature extraction seam

## Constraints
- Patch metadata must stay independent of feature extractor choice.
- Optional deep-feature code must fail with an actionable import error if Torch
  dependencies are unavailable.
- Tests must remain fast and deterministic.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`

## Acceptance criteria
- Existing tests continue to pass.
- A custom extractor can build and score through the same PatchCore path.
- The optional Torch extractor can be imported only when dependencies exist.
- Documentation clearly says the deep extractor path is optional and not yet the
  final PatchCore model.

## Risks
- Torch/Torchvision versions are large and platform-sensitive, so this task
  should define the integration point before adding those dependencies.
- The feature protocol may need to evolve once multi-scale feature maps are
  introduced.

## Progress log
### 2026-04-15
- Completed: patch feature extractor protocol, mean RGB extractor implementation,
  lazy optional Torch/Torchvision wrapper, refactored memory-bank/scoring path,
  tests, docs, and local MVTec smoke check.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`, and
  `./.venv/bin/pytest -q` passed. A local smoke check against the prepared
  MVTec AD bottle manifest built 392 mean-RGB patch features from 2 training
  images and scored 196 query patches through the refactored default path.
- Remaining: implement the concrete Torch/Torchvision backbone extractor,
  pretrained-weight policy, and deep feature caching in a follow-up task.
