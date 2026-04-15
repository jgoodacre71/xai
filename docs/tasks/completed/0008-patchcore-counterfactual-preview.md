# 0008-patchcore-counterfactual-preview: patch replacement counterfactual preview

## Status
Complete

## Owner
Codex thread

## Why
REPO_SPEC.md section 3.2 lists a counterfactual patch replacement preview as an
optional PatchCore hero view, and section 8.3 recommends replacing the top
anomalous patch with the nearest normal patch and recomputing the score. The
current bottle report shows evidence and provenance, but not what would change
if the top patch were replaced by its nearest normal exemplar.

## Source of Truth
- AGENTS.md
- .agents/PLANS.md
- REPO_SPEC.md sections 3.2 and 8.3
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md
- docs/tasks/completed/0007-patchcore-coarse-anomaly-map.md

## Scope
- Add reusable image-level patch replacement helper.
- Add score recomputation for the replaced image preview.
- Add a `CounterfactualArtefact` for the report path.
- Add the counterfactual preview image and score delta to the bottle report.
- Add tests with synthetic images.

## Out of Scope
- Causal claims.
- Seamless inpainting.
- Multiple patch replacement search.
- Notebook work.

## Deliverables
- counterfactual helper under `src/xai_demo_suite/explain/`
- report update showing the replacement preview and score change
- tests for replacement coordinates and score delta
- docs update noting limitations

## Constraints
- Preserve source patch provenance: replacement patch must come from the recorded
  nearest-normal source image path and source box.
- Keep generated outputs ignored by git.
- Make clear this is a didactic probe, not causal proof.
- Keep package code reusable by future notebooks.

## Proposed File Changes
- `src/xai_demo_suite/explain/counterfactuals.py`
- `src/xai_demo_suite/reports/patchcore_bottle.py`
- `tests/unit/test_patchcore_report.py`
- `docs/PATCHCORE_NOTES.md`

## Validation Plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --no-cache`

## Acceptance Criteria
- Report writes a counterfactual replacement preview asset.
- Report displays before score, after score, and score delta.
- Tests prove the replacement uses recorded query and source coordinates.
- Tests prove a `CounterfactualArtefact` is created for the preview.
- Full checks and local report smoke command pass.

## Risks
- Patch replacement can create visual seams; the report should not overclaim.
- Random-weight ResNet features make the score delta illustrative until the
  final feature policy is chosen.

## Decision Log
### 2026-04-15
- Decision: implement nearest-normal patch replacement before notebooks.
- Reason: AGENTS.md requires reusable source code first, and the spec explicitly
  recommends this PatchCore probe.
- Follow-up: add a notebook section later that interprets this generated report.

## Progress Log
### 2026-04-15
- Completed: task created after audit; patch replacement helper implemented;
  report integrated with counterfactual preview and score delta; tests and docs
  updated.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`,
  `./.venv/bin/pytest -q`, and
  `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --no-cache`
  passed.
- Remaining: future work should add stronger interpretation text once the final
  feature policy is chosen.
