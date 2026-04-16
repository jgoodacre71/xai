# 0032: MVTec AD 2 Support

## Status
Complete

## Owner
Codex thread

## Why
The first-wave suite is complete, and the next second-wave expansion from the
spec is MVTec AD 2. It keeps the repo focused on industrial anomaly detection
while adding a stronger future path for robustness and acquisition-shift work.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DATASETS.md
- data_registry.yaml

## Scope
- Add local dataset support for MVTec AD 2.
- Extend the dataset CLI to list, fetch, and prepare the dataset.
- Build canonical processed manifests per discovered scenario.
- Update repo docs and task memory.

## Out of scope
- Wiring MVTec AD 2 into a generated report in the same task.
- Hard-coding a brittle direct archive link from the official site.
- Implementing VisA, MetaShift, or Spawrious in the same pass.

## Deliverables
- `src/xai_demo_suite/data/downloaders/mvtec_ad_2.py`
- CLI support in `src/xai_demo_suite/cli/data.py`
- Tests for downloader and CLI behaviour
- Updated dataset docs, registry, TODOs, and task memory

## Constraints
- Keep the fetch flow honest: use the official source page and allow explicit
  direct URLs or manual archive placement instead of assuming a stable hidden
  archive link.
- Keep raw archives and extracted data uncommitted.
- Use UK English in docs and user-facing text.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_mvtec_ad_2_downloader.py -q`
4. `./.venv/bin/xai-demo-data list`
5. `./.venv/bin/xai-demo-data fetch mvtec_ad_2 --category all --dry-run`

## Risks
- The official page is the stable source of truth, but direct archive links may
  not be stable or may be access-gated.
- The extracted dataset structure may evolve, so the manifest builder needs to
  discover scenario folders and supported split names defensively.

## Decision log
### 2026-04-16
- Decision: Implement MVTec AD 2 as the first second-wave expansion.
- Reason: It stays close to the current anomaly-detection focus and supports
  future robustness work without widening the repo into a new task family yet.

### 2026-04-16
- Decision: Do not hard-code a direct dataset archive URL.
- Reason: The official source page is stable, but direct archive links are more
  brittle; an explicit `--archive-url` or manual archive placement is safer.

## Progress log
### 2026-04-16
- Completed: Added MVTec AD 2 metadata, fetch planning, extraction, scenario
  discovery, and per-scenario manifest writing.
- Completed: Extended the dataset CLI with `mvtec_ad_2` list, fetch, and
  prepare paths, including explicit `--archive-url` and `--archive-path`
  handling.
- Completed: Updated dataset docs, the registry, TODOs, and demo status notes.
- Completed: Added focused downloader and CLI tests for the new dataset path.
