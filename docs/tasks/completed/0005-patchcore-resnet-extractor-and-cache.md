# 0005-patchcore-resnet-extractor-and-cache: ResNet extractor and feature cache

## Status
Complete

## Owner
Codex thread

## Why
The PatchCore path now has a feature extractor interface, but the only concrete
extractor is mean RGB. The hero demo needs a real deep-feature path and cached
artefacts so local runs can be repeated without recomputing the same patch
features.

## Source docs
- REPO_SPEC.md
- docs/PATCHCORE_NOTES.md
- docs/ARCHITECTURE.md
- docs/DATASETS.md
- docs/tasks/completed/0004-patchcore-feature-extractor-interface.md
- PyTorch official install guidance

## Scope
- Add an optional `ml` dependency group for Torch and Torchvision.
- Implement `TorchvisionBackbonePatchFeatureExtractor` for ResNet patch crops.
- Keep pretrained weights optional and never download weights implicitly.
- Add memory-bank save/load helpers under the PatchCore package.
- Add tests for cache round-trips and optional deep extractor behaviour.
- Run a small local smoke check against MVTec AD bottle.

## Out of scope
- Coreset selection.
- Multi-scale PatchCore feature maps.
- Full anomaly-map rendering.
- Notebook/demo page rendering.
- Downloading pretrained weights.

## Deliverables
- concrete Torch/Torchvision extractor implementation
- `src/xai_demo_suite/models/patchcore/cache.py`
- tests for cache and deep extractor shape
- docs update for optional ML setup

## Constraints
- Raw data and cached artefacts must remain ignored.
- Base tests must still run without requiring Torch if the `ml` group is absent.
- The deep extractor must not download pretrained weights by default.
- Provenance metadata must survive save/load.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. Small local MVTec bottle smoke check with `weights_name=None`.

## Acceptance criteria
- A ResNet extractor returns one vector per patch.
- Cache round-trip preserves feature rows and patch metadata.
- The mean-RGB path remains unchanged for lightweight runs.
- The local smoke check builds and reloads a deep-feature memory-bank artefact
  under ignored `data/artefacts/`.

## Risks
- Torch/Torchvision are large dependencies and may behave differently across
  platforms.
- Patch-crop features are a practical bridge, not the final multi-scale
  PatchCore feature-map implementation.

## Progress log
### 2026-04-15
- Completed: optional ML dependency group, concrete ResNet-18 patch-crop
  extractor, memory-bank cache save/load helpers, tests, docs, and local MVTec
  smoke check.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`, and
  `./.venv/bin/pytest -q` passed. A local MVTec AD bottle smoke check built 49
  ResNet feature vectors from one nominal image, saved/reloaded the cache at
  `data/artefacts/patchcore/bottle/resnet18_smoke_bank.npz`, and scored 49 query
  patches with nearest-normal provenance.
- Remaining: add coreset selection, anomaly-map rendering, and a thin demo
  runner/visual report.
