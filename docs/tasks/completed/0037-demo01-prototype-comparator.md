# 0037: Demo 01 Prototype Comparator

## Status
Complete

## Owner
Codex thread

## Why
The remaining unchecked TODO from the spec was to add an interpretable
comparator where it genuinely improves a demo. Demo 01 was the right place: a
prototype-style comparator can expose which training exemplars a crossed-group
sample resembles without adding another unrelated model family.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DEMO_STATUS.md
- docs/DEMO_CATALOGUE.md
- docs/tasks/completed/0036-demo01-metashift-extension.md

## Scope
- Add a prototype-style exemplar comparator on top of the frozen feature path in
  Demo 01.
- Show comparator metrics and nearest training exemplars in the real-data
  report.
- Reuse the same path for the optional MetaShift extension when prepared.
- Update tests, docs, task memory, and TODOs.

## Out of scope
- Full ProtoPNet training.
- A new standalone report.
- Applying the comparator to unrelated demos in the same pass.

## Deliverables
- Updated `src/xai_demo_suite/models/classification/waterbirds.py`
- Updated `src/xai_demo_suite/reports/waterbirds_shortcut.py`
- Focused tests
- Updated docs and task memory

## Constraints
- Keep the comparator narrow and legible.
- Use the existing frozen backbone rather than introducing another heavy
  training dependency.
- Keep report sections compact enough to remain demo-readable.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_waterbirds_shortcut_report.py -q`
4. `./.venv/bin/xai-demo-report waterbirds-shortcut --no-real-data`

## Risks
- The report could get crowded if comparator evidence were not limited to a few
  useful exemplars.
- A comparator that was too weak or too opaque would not justify closing the
  remaining TODO.

## Decision log
### 2026-04-16
- Decision: Implement a prototype-exemplar comparator instead of full ProtoPNet.
- Reason: It delivers interpretable nearest-example evidence for the shortcut
  story without bloating the repo with another heavy model family.

## Progress log
### 2026-04-16
- Completed: Added a prototype-exemplar comparator over frozen image embeddings
  with class-prototype prediction, per-group metrics, and nearest training
  exemplar retrieval.
- Completed: Extended Demo 01 so both the Waterbirds slice and the optional
  MetaShift extension can show comparator metrics, prototype-margin sensitivity
  to masking, and nearest predicted versus contrast exemplars.
- Completed: Updated tests, README, demo status, demo catalogue, TODOs, and
  task memory.
