# 0019: Demo Suite Runner and Verifier

## Status
Complete

## Owner
Codex thread

## Why
The repository now has multiple working demos. To make the suite feel like a
product rather than a set of isolated scripts, it needs one command to rebuild
the demo suite and one command to verify that generated reports, demo cards,
figures, and the local index are coherent.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/DEMO_CATALOGUE.md

## Scope
Add reusable suite-generation and verification helpers, expose them through
`xai-demo-report`, update docs, and add tests.

## Out of scope
- Uploading to GitHub.
- Packaging generated outputs for release.
- Executing notebooks.

## Deliverables
- Suite build helper for current local reports.
- Output verification helper for demo cards, figures, report links, and index.
- CLI commands.
- Tests and README/docs updates.

## Constraints
- Raw data and generated outputs remain uncommitted.
- The synthetic suite should work without external datasets.
- MVTec report generation should be opt-in for fresh clones without local data.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/reports/suite.py`
- `src/xai_demo_suite/cli/demo.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report suite --include-mvtec --no-cache`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- The MVTec hero demo depends on local data, so the suite command must make
  that dependency explicit.

## Decision log
### 2026-04-15
- Decision: Make the default suite command generate synthetic demos only, with
  `--include-mvtec` for the local MVTec bottle report.
- Reason: Fresh clones should have a working demo suite without external data,
  while this machine can still regenerate the full local set.
- Follow-up: Add release packaging later.

## Progress log
### 2026-04-15
- Completed: Audited repo, git memory, spec, agents, and existing outputs.
- Verification: Git status was clean before edits.
- Remaining: Implement, test, generate, verify, and commit.

### 2026-04-15
- Completed: Added suite build and verification helpers, CLI commands, unit
  tests, README workflow notes, and current demo status documentation.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report suite --include-mvtec
  --no-cache`; `./.venv/bin/xai-demo-report verify`.
- Remaining: Package or publish generated demos later once a GitHub remote and
  release workflow are configured.
