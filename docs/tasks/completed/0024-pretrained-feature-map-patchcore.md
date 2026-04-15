# 0024: Pretrained Feature-Map PatchCore

## Status
Complete

## Owner
Codex thread

## Why
The current PatchCore demos preserve provenance and are honest, but they still
use deterministic colour/texture patch features by default. The spec calls for
serious industrial anomaly demos, which means adding a proper deep feature-map
PatchCore path with explicit pretrained-weight handling and memory-bank
reduction.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md
- docs/DEMO_STATUS.md

## Scope
Add a Torchvision dense feature-map extractor, add deterministic coreset memory
reduction, expose the model through Demo 03 CLI/config choices, and keep
deterministic fallbacks for tests and fresh clones.

## Out of scope
- Full benchmark reproduction against official PatchCore numbers.
- GPU-specific optimisation.
- Configuring GitHub remote.
- Replacing every synthetic demo with real data in this task.

## Deliverables
- Dense Torchvision feature-map extractor with source-patch provenance intact.
- Coreset memory-bank reduction helper.
- Demo 03 CLI choices for random/pretrained feature-map PatchCore.
- Tests for feature-map extraction and coreset reduction.
- Docs/status updates.

## Constraints
- Pretrained weights must be explicit, not downloaded silently.
- Raw data, model caches, and generated artefacts remain uncommitted.
- Reports must clearly distinguish deterministic, random, and pretrained paths.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/models/patchcore/features.py`
- `src/xai_demo_suite/models/patchcore/baseline.py`
- `src/xai_demo_suite/models/patchcore/__init__.py`
- `src/xai_demo_suite/reports/patchcore_bottle.py`
- `src/xai_demo_suite/cli/demo.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_random --max-train 4 --max-examples 1 --no-cache`
5. Optionally, if weights resolve locally or can download: `./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_pretrained --max-train 4 --max-examples 1 --no-cache`
6. `./.venv/bin/xai-demo-report verify`

## Risks
- Pretrained weight download may be unavailable in some environments.
- Dense feature extraction is slower than deterministic colour/texture features.
- The first implementation should be strong and inspectable, while remaining
  bounded enough to run locally.

## Decision log
### 2026-04-15
- Decision: Add ResNet-18 feature-map PatchCore first, with random and
  pretrained choices.
- Reason: It gives a real deep-feature path quickly, keeps runtime manageable on
  CPU, and leaves WideResNet50-2 as a later performance/quality upgrade.
- Follow-up: Add WideResNet50-2 multi-layer features and benchmark metrics.
- Decision: Default report cache paths are now extractor-specific, and coreset
  runs include the requested coreset size in the generated cache filename.
- Reason: A serious demo suite should not silently reuse a memory bank built
  with a different extractor or retained-bank size.
- Follow-up: Store richer memory-bank build metadata inside cache artefacts if
  the cache format is revised later.

## Progress log
### 2026-04-15
- Completed: Revisited repo state, agents, PatchCore notes, Demo status, and
  existing feature extractor/report code.
- Verification: Git working tree was clean before edits.
- Completed: Added dense Torchvision ResNet-18 feature-map extraction, greedy
  k-centre coreset reduction, Demo 03 CLI choices for random/pretrained feature
  maps, extractor-specific caches, report run-context text, and tests.
- Verification:
  `./.venv/bin/ruff check .`;
  `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`;
  `./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_random --max-train 4 --max-examples 1 --coreset-size 64 --input-size 128 --no-cache`;
  `./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_pretrained --max-train 20 --max-examples 3 --coreset-size 512 --input-size 224 --no-cache`;
  `./.venv/bin/xai-demo-report verify`.
- Remaining: Official benchmark metrics, WideResNet50-2 quality pass, and
  component-aware comparators remain later tasks.
