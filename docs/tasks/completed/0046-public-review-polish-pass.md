# 0046 - Public review polish pass

## Context

The repo had become a serious local demo suite, but the public review surfaced
four concrete weaknesses:

- the review pack was noisy and duplicated demo variants;
- the local hub and the review pack disagreed on dataset-readiness coverage;
- the docs still described older PatchCore and Demo 01 states in places;
- public CI did not exercise the synthetic report build and verification path.

There was also a presentational gap in the strongest demo pages:

- Demo 02 still read like a worksheet rather than a compact selected-case story;
- Demo 03 had the right ingredients, but still needed tighter nominal-context
  framing around score scale and counterfactual evidence.

## Goal

Tighten the repo so the public GitHub branch presents one clearer story:

1. a curated review layer with no accidental duplicate cards;
2. consistent dataset-readiness reporting;
3. docs that match the actual current implementation state;
4. public CI proof that synthetic reports still build and verify;
5. stronger Demo 02 and Demo 03 presentation without changing the core claims.

## What changed

- Expanded the shared prepared-dataset list in
  [cards.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/cards.py)
  so the local hub now tracks the same broader dataset set as the review pack.
- Reworked
  [review_pack.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/review_pack.py)
  to:
  - parse build metadata from demo cards,
  - curate one primary page per demo,
  - group Demo 03 variants separately,
  - report build coherence,
  - use the stronger current walkthrough order.
- Updated
  [docs/REVIEW_GUIDE.md](/Users/johngoodacre/work/xai/docs/REVIEW_GUIDE.md)
  so the review order and Demo 01 language match the current outputs.
- Updated
  [docs/PATCHCORE_NOTES.md](/Users/johngoodacre/work/xai/docs/PATCHCORE_NOTES.md)
  so the “current implementation status” reflects the serious feature-map
  PatchCore path rather than only the original deterministic baseline.
- Added a second public CI job in
  [.github/workflows/ci.yml](/Users/johngoodacre/work/xai/.github/workflows/ci.yml)
  that builds the synthetic suite via installed console scripts and runs
  `xai-demo-report verify`.
- Tightened
  [shortcut_industrial.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/shortcut_industrial.py)
  with a selected-case matrix and a less table-first layout.
- Tightened
  [patchcore_bottle.py](/Users/johngoodacre/work/xai/src/xai_demo_suite/reports/patchcore_bottle.py)
  with clearer counterfactual framing and split-level score context.
- Extended focused tests in:
  - [test_review_pack.py](/Users/johngoodacre/work/xai/tests/unit/test_review_pack.py)
  - [test_patchcore_report.py](/Users/johngoodacre/work/xai/tests/unit/test_patchcore_report.py)
  - [test_shortcut_industrial_report.py](/Users/johngoodacre/work/xai/tests/unit/test_shortcut_industrial_report.py)

## Validation

1. `./.venv/bin/ruff check src tests .github/workflows`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_review_pack.py tests/unit/test_demo_cards.py tests/unit/test_patchcore_report.py tests/unit/test_shortcut_industrial_report.py tests/unit/test_suite_reports.py tests/integration/test_cli_end_to_end.py -q -ra`

## Outcome

The repo now has a much cleaner public review surface:

- one curated review-pack flow instead of a flat dump of local card folders;
- one broader, consistent dataset-readiness story;
- public CI proof for the synthetic report-generation path;
- docs that better match the current state of the demos;
- stronger flagship presentation on Demo 02 and Demo 03.

The remaining step after this task is to rebuild the committed HTML snapshot on
one coherent code SHA and publish it.
