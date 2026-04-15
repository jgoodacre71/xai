# 0023: Real MVTec LOCO Demo 07

## Status
Complete

## Owner
Codex thread

## Why
The suite now has all eight demo slots, but Demo 07 is still a synthetic proxy.
The spec explicitly calls for MVTec LOCO AD to show that local patch novelty is
not the same as logical understanding.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/DATASETS.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0022-mvtec-loco-data-sourcing.md

## Scope
Fetch and prepare one local MVTec LOCO AD category, then upgrade Demo 07 so it
can use real LOCO structural/logical examples when the manifest is present while
retaining the synthetic fallback for fresh clones.

## Out of scope
- Committing raw or processed dataset files.
- Downloading all LOCO categories.
- Adding a full logic-aware model.
- Configuring a GitHub remote.

## Deliverables
- Local MVTec LOCO `juice_bottle` archive and manifest, ignored by git.
- Real-data Demo 07 report path using LOCO examples when available.
- CLI and suite behaviour that remains robust for fresh clones.
- Tests for report selection/fallback behaviour.
- Docs/status updates.

## Constraints
- Never commit raw data, extracted data, processed manifests, or generated
  caches.
- Preserve PatchCore source image and patch-coordinate provenance.
- Keep report text honest about PatchCore-style local novelty versus logical
  rule understanding.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/reports/patchcore_logic.py`
- tests for real/fallback Demo 07 behaviour
- docs/status/catalogue/notes updates

## Validation plan
1. `./.venv/bin/xai-demo-data fetch mvtec_loco_ad --category juice_bottle`
2. `./.venv/bin/xai-demo-data prepare mvtec_loco_ad --category juice_bottle`
3. `./.venv/bin/ruff check .`
4. `./.venv/bin/mypy src`
5. `./.venv/bin/pytest -q`
6. `./.venv/bin/xai-demo-report patchcore-logic --no-cache`
7. `./.venv/bin/xai-demo-report suite --include-mvtec --no-cache`
8. `./.venv/bin/xai-demo-report verify`

## Risks
- LOCO download may be slow or the official archive URL may fail.
- The first real report should be compelling but still bounded; a full
  component-aware comparator remains later work.

## Decision log
### 2026-04-15
- Decision: Use `juice_bottle` first.
- Reason: It is the smallest LOCO category listed in the current registry and
  is enough to replace the synthetic-only logical-anomaly report path.
- Follow-up: Add a second category later if the first one is not visually strong
  enough.

## Progress log
### 2026-04-15
- Completed: Revisited repo status, agents, demo status, and LOCO sourcing
  memory.
- Verification: Git working tree was clean before edits.
- Remaining: Fetch, prepare, implement real-data report path, test, regenerate,
  verify, and commit.

### 2026-04-15
- Completed: Downloaded and prepared MVTec LOCO AD `juice_bottle`; corrected the
  LOCO mask lookup for nested sample-id mask folders; upgraded Demo 07 to use
  real LOCO logical/structural examples when the manifest exists; retained the
  synthetic fallback for fresh clones; added tests for both paths.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report patchcore-logic
  --no-cache`; `./.venv/bin/xai-demo-report verify`.
- Remaining: Regenerate the full suite, finalise docs, and commit.

### 2026-04-15
- Completed: Regenerated the full local suite with MVTec AD and MVTec LOCO
  present, verified demo-card integrity, and confirmed LOCO raw/interim/processed
  data paths are ignored by git.
- Verification: `./.venv/bin/xai-demo-report suite --include-mvtec --no-cache`;
  `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`; `./.venv/bin/pytest -q`;
  `./.venv/bin/xai-demo-report verify`.
- Remaining: Add component-aware/OCR comparison and pretrained multi-scale
  PatchCore in later tasks.
