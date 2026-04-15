# 0003-patchcore-provenance-foundation: PatchCore provenance foundation

## Status
Complete

## Owner
Codex thread

## Why
PatchCore is the hero demo, and the repository specification makes source patch
provenance non-negotiable. Before adding heavy feature extractors or notebooks,
we need a small, testable implementation that stores nominal patch metadata,
scores test patches against a memory bank, and returns real nearest-normal
source evidence.

## Source docs
- REPO_SPEC.md
- docs/ARCHITECTURE.md
- docs/PATCHCORE_NOTES.md
- docs/XAI_CONTRACT.md
- docs/DATASETS.md
- docs/decisions/ADR-0003-explanation-contract.md

## Scope
- Add typed PatchCore-style patch metadata and nearest-neighbour result objects.
- Build a simple baseline memory bank from nominal image patches with source
  image ids and patch coordinates.
- Score candidate images by patch distance to the nominal bank.
- Convert nearest-normal evidence into the shared provenance artefact contract.
- Add tests using synthetic image fixtures and the local MVTec AD manifest shape.

## Out of scope
- Deep neural feature extraction.
- Coreset selection.
- Full anomaly-map visualisation.
- Notebook and report rendering.
- Downloading additional dataset categories.

## Deliverables
- `src/xai_demo_suite/models/patchcore/`
- `src/xai_demo_suite/data/manifests.py`
- tests for manifest loading, patch metadata, scoring, and provenance artefacts
- docs update describing the baseline PatchCore slice

## Constraints
- Preserve source image ids and patch coordinates for every stored patch.
- Keep scoring and provenance display data separate but explicitly linked.
- Do not commit raw data or generated artefacts.
- Keep first implementation deterministic and fast.

## Proposed file changes
- `src/xai_demo_suite/data/manifests.py` for JSONL manifest loading.
- `src/xai_demo_suite/models/patchcore/types.py` for typed records.
- `src/xai_demo_suite/models/patchcore/baseline.py` for simple patch features,
  memory-bank construction, scoring, and provenance conversion.
- `tests/unit/test_mvtec_manifest.py` and
  `tests/unit/test_patchcore_baseline.py` for regression coverage.
- `docs/PATCHCORE_NOTES.md` for current implementation notes.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`

## Acceptance criteria
- Tests prove every retained nominal patch has source image id and coordinates.
- Scoring returns nearest normal patch evidence for anomalous patches.
- The shared `ProvenanceArtefact` can be built from scoring output.
- Local MVTec AD data is not required for tests, but the code can read the
  prepared manifest when present.

## Risks
- A handcrafted patch-feature baseline is not full PatchCore; it is a foundation
  for provenance and API shape only.
- Patch stride/size choices may need revision once deep features are added.

## Progress log
### 2026-04-15
- Completed: manifest loading, mean-RGB PatchCore-style memory bank, patch
  source metadata, nearest-normal scoring, provenance artefact conversion,
  tests, docs, and local MVTec smoke check.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`, and
  `./.venv/bin/pytest -q` passed. A local smoke check against the prepared
  MVTec AD bottle manifest built 588 nominal patch features from 3 training
  images and scored 196 query patches.
- Remaining: replace mean-RGB features with deep features, add anomaly-map
  rendering, coreset support, and patch counterfactual probes.
