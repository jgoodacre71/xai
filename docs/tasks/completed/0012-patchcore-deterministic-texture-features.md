# 0012: PatchCore Deterministic Texture Features

## Status
Complete

## Owner
Codex thread

## Why
The mask localisation check added in task 0011 shows that the current
random-weight ResNet patch-crop report can produce visually poor localisation
on the first selected MVTec AD bottle examples. The hero PatchCore demo needs a
stronger reproducible feature path that works without implicit network
downloads or uncommitted model weights.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md

## Scope
Evaluate current local feature choices, add a stronger deterministic patch
feature extractor if useful, and expose report extractor selection through the
CLI.

## Out of scope
- Downloading pretrained ImageNet weights.
- Implementing full multi-scale feature-map PatchCore.
- Benchmark-grade model selection.
- Committing generated data, caches, or reports.

## Deliverables
- A deterministic patch feature extractor suitable for local demo runs.
- CLI/config selection between available report extractors.
- Tests for feature extraction and report configuration.
- Regenerated local report showing measured mask checks.
- Documentation updates describing the extractor honestly.

## Constraints
- Do not implicitly download pretrained weights.
- Preserve source patch provenance and coordinates.
- Keep the mean RGB baseline available for deterministic tests.
- Keep generated artefacts ignored by git.

## Proposed file changes
- `src/xai_demo_suite/models/patchcore/features.py` — add feature extractor.
- `src/xai_demo_suite/reports/patchcore_bottle.py` — report extractor
  selection and run-context copy.
- `src/xai_demo_suite/cli/demo.py` — expose extractor option.
- `tests/unit/test_patchcore_baseline.py` and/or
  `tests/unit/test_patchcore_report.py` — cover new behaviour.
- `README.md` and `docs/PATCHCORE_NOTES.md` — document the path.

## Validation plan
1. Measure mask overlap for current and candidate feature choices.
2. `./.venv/bin/ruff check .`
3. `./.venv/bin/mypy src`
4. `./.venv/bin/pytest -q`
5. `./.venv/bin/xai-demo-report patchcore-bottle --max-examples 3 --no-cache`

## Risks
- Hand-crafted features may improve visible localisation but still are not the
  final academic PatchCore model.
- More patches and more training images can make local generation slower.
- Good top-patch overlap on selected examples does not prove benchmark quality.

## Decision log
### 2026-04-15
- Decision: Prefer a deterministic, dependency-light feature upgrade before
  pretrained deep features.
- Reason: Network is restricted and the repo should remain reproducible from
  checked-in code and local data.
- Follow-up: Add proper pretrained feature-map PatchCore once weight sourcing
  and caching policy are explicit.

## Progress log
### 2026-04-15
- Completed: Opened task after task 0011 exposed weak random-weight
  localisation.
- Verification: Previous task checks were clean.
- Remaining: Measure, implement, regenerate, and commit.

### 2026-04-15
- Completed: Measured local feature candidates, added the deterministic
  `colour_texture` extractor, made it the report default, exposed
  `--feature-extractor`, and updated docs/tests.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report patchcore-bottle --max-examples 3 --no-cache`.
- Remaining: None for this task. The regenerated report now shows mask
  intersection for the first three selected examples, but this is still a
  local deterministic feature path rather than full pretrained PatchCore.
