# 0045: Repo Hardening from Review Feedback

## Status
Completed

## Why
External review surfaced a real set of credibility gaps: packaging guidance was
inconsistent, the repo had no visible CI, integration coverage was missing, the
suite verifier was mostly checking file existence, and notebook smoke coverage
did not extend across the full demo set.

## Changes
- Added `project.optional-dependencies` in
  [pyproject.toml](/Users/johngoodacre/work/xai/pyproject.toml) so `pip`
  install instructions now match the documented extras for `dev`, `notebooks`,
  and `ml`.
- Updated [README.md](/Users/johngoodacre/work/xai/README.md) and
  [AGENTS.md](/Users/johngoodacre/work/xai/AGENTS.md) so the install and
  validation story is consistent:
  - `uv` is the preferred workflow,
  - `.venv`/`pip` is the explicit fallback,
  - optional ML setup is documented correctly for both.
- Added a public CI workflow at
  [.github/workflows/ci.yml](/Users/johngoodacre/work/xai/.github/workflows/ci.yml)
  covering `ruff`, `mypy`, and `pytest`.
- Added end-to-end CLI integration coverage in
  [test_cli_end_to_end.py](/Users/johngoodacre/work/xai/tests/integration/test_cli_end_to_end.py)
  for:
  - `xai_demo_report suite`
  - `xai_demo_report verify`
  - `xai_demo_report review-pack`
  - `xai_demo_data list`
- Strengthened
  [verify_demo_suite_outputs](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/suite.py)
  so it now checks:
  - required demo-card fields,
  - non-empty semantic card content,
  - expected report HTML markers by demo,
  - local index markers,
  - review-pack markers when present.
- Broadened notebook smoke coverage in
  [test_notebook_smoke.py](/Users/johngoodacre/work/xai/tests/unit/test_notebook_smoke.py)
  to cover Demo 03, 05, 06, and 07 as well as the already-smoked notebook
  scripts.
- Updated [docs/DEMO_STATUS.md](/Users/johngoodacre/work/xai/docs/DEMO_STATUS.md)
  so the public status page reflects the stronger verifier and the new CI /
  integration proof surface.

## Validation
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_suite_reports.py tests/unit/test_notebook_smoke.py tests/integration/test_cli_end_to_end.py -q`
4. `./.venv/bin/xai-demo-report verify`
