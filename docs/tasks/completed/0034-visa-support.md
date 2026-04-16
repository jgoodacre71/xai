# 0034: VisA Support

## Status
Complete

## Owner
Codex thread

## Why
The first second-wave dataset step was in place with MVTec AD 2. The next gap
in the spec and TODOs was VisA, which adds another real anomaly-detection
dataset family without changing the core four-pillar structure.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DATASETS.md
- data_registry.yaml

## Scope
- Add local dataset support for the VisA one-class split.
- Extend the dataset CLI to list, fetch, and prepare VisA.
- Build canonical processed manifests per prepared category.
- Update docs, registry metadata, and task memory.

## Out of scope
- Wiring VisA into a generated report in the same task.
- Adding MetaShift or Spawrious in the same pass.
- Reworking the existing MVTec-based demos.

## Deliverables
- `src/xai_demo_suite/data/downloaders/visa.py`
- CLI support in `src/xai_demo_suite/cli/data.py`
- Focused downloader and CLI tests
- Updated docs, registry metadata, TODOs, and task memory

## Constraints
- Keep raw archives, split CSV files, extracted data, and manifests
  uncommitted.
- Record the upstream source and licence clearly.
- Keep the canonical processed output aligned with the rest of the repo:
  `data/processed/visa/<category>/manifest.jsonl`.
- Use UK English in docs and user-facing text.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_visa_downloader.py -q`
4. `./.venv/bin/xai-demo-data list`
5. `./.venv/bin/xai-demo-data fetch visa --category all --dry-run`

## Risks
- The upstream split CSV schema may change, so the preparer needs to fail
  clearly if required columns disappear.
- Raw archive layout may vary slightly from the expected top-level folder name.

## Decision log
### 2026-04-16
- Decision: Implement VisA support around the published one-class split CSV.
- Reason: That yields a clean anomaly-detection adapter and keeps the processed
  layout aligned with the rest of the repo.

### 2026-04-16
- Decision: Resolve prepared anomaly masks by exact filename first and by stem
  second.
- Reason: The prepared VisA mask filenames do not need to match the image
  suffix exactly, so the manifest builder should tolerate `.png` masks for
  `.JPG` images.

## Progress log
### 2026-04-16
- Completed: Added VisA dataset metadata, archive and split-CSV fetch helpers,
  extraction, one-class layout preparation, and per-category manifest writing.
- Completed: Extended the dataset CLI with `visa` list, fetch, and prepare
  paths.
- Completed: Updated dataset docs, registry metadata, TODOs, README notes, and
  demo-status notes.
- Completed: Added focused downloader and CLI tests, including coverage for the
  prepared mask resolution path.
