# 0044: Review Guide and ChatGPT Handoff

## Status
Completed

## Why
The repo already had strong local outputs, but it still lacked one durable
reviewer document that explained the whole build-out, the logic of each demo,
and the cleanest way to hand the work to ChatGPT or another external reviewer.

## Changes
- Added [REVIEW_GUIDE.md](/Users/johngoodacre/work/xai/docs/REVIEW_GUIDE.md)
  as a repo-level reviewer document covering:
  - what was built throughout,
  - the best local review order,
  - the best current GitHub-versus-Projects path for ChatGPT review,
  - suggested ChatGPT prompts,
  - a demo-by-demo interpretation guide.
- Upgraded [review_pack.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/review_pack.py)
  so the generated review pack includes:
  - a recommended walkthrough order,
  - direct links to the new review guide and core repo docs,
  - a more specific ChatGPT handoff section.
- Updated [README.md](/Users/johngoodacre/work/xai/README.md) to surface the
  review guide and review-pack entry points.
- Extended [test_review_pack.py](/Users/johngoodacre/work/xai/tests/unit/test_review_pack.py)
  to cover the stronger reviewer-facing copy.

## Validation
1. `./.venv/bin/pytest tests/unit/test_review_pack.py -q`
2. `./.venv/bin/ruff check src tests`
3. `./.venv/bin/mypy src`
4. `./.venv/bin/xai-demo-report review-pack`
5. `./.venv/bin/xai-demo-report verify`
