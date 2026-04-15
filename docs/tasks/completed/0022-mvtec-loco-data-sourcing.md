# 0022: MVTec LOCO AD Data Sourcing

## Status
Complete

## Owner
Codex thread

## Why
The spec requires MVTec LOCO AD for logical anomaly limitations. The suite now
has a synthetic Demo 07 proxy, but the repository still lacks a first-class LOCO
fetch/prepare workflow and metadata entry.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/DATASETS.md
- data_registry.yaml
- MVTec LOCO AD official downloads page

## Scope
Add MVTec LOCO AD category metadata, safe raw-archive fetch planning, archive
extraction, manifest generation for train/validation/test images, CLI support,
tests, and docs/registry updates.

## Out of scope
- Downloading the full LOCO dataset during this task.
- Replacing Demo 07 with real LOCO examples.
- Implementing a logic-aware comparator.

## Deliverables
- `src/xai_demo_suite/data/downloaders/mvtec_loco_ad.py`
- `xai-demo-data fetch mvtec_loco_ad ...`
- `xai-demo-data prepare mvtec_loco_ad ...`
- Tests using small fixture archives.
- Dataset docs and registry metadata.

## Constraints
- Raw archives and extracted data remain uncommitted.
- Never overwrite existing raw archives unless `--overwrite` is explicit.
- Extract archives into `data/interim/`, not `data/raw/`.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/data/downloaders/mvtec_loco_ad.py`
- `src/xai_demo_suite/cli/data.py`
- `data_registry.yaml`
- `docs/DATASETS.md`
- tests

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-data list`
5. `./.venv/bin/xai-demo-data fetch mvtec_loco_ad --category juice_bottle --dry-run`

## Risks
- Official direct archive URLs may change; the registry should still point back
  to the official MVTec downloads page as the durable source of truth.

## Decision log
### 2026-04-15
- Decision: Implement the same safe fetch/prepare pattern as MVTec AD.
- Reason: It already matches the repository data policy and keeps raw archives
  isolated from processed manifests.
- Follow-up: Use the prepared LOCO manifest to add real examples to Demo 07.

## Progress log
### 2026-04-15
- Completed: Checked official MVTec LOCO downloads page and current MVTec AD
  downloader implementation.
- Verification: Git working tree was clean before edits.
- Remaining: Implement, test, document, and commit.

### 2026-04-15
- Completed: Added MVTec LOCO AD metadata, category archive URLs, safe
  fetch/prepare helpers, manifest generation, CLI support, fixture-archive
  tests, registry entries, and dataset documentation.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-data list`;
  `./.venv/bin/xai-demo-data fetch mvtec_loco_ad --category juice_bottle
  --dry-run`; `./.venv/bin/xai-demo-report verify`.
- Remaining: Download one LOCO category locally and replace the synthetic Demo
  07 proxy with real structural/logical examples.
