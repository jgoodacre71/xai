# Active Context

## Active surface on `main`

The active working surface is the curated notebook-first demo suite built
around:

- Demo 01: Waterbirds shortcut
- Demo 02: industrial shortcut trap
- Demo 03: PatchCore on MVTec AD bottle
- Demo 04: PatchCore learns the wrong normal
- Demo 05: PatchCore limits lab
- Demo 06: PatchCore severity mismatch
- Demo 07: PatchCore logical anomaly limits
- Demo 08: explanation drift under shift

The primary repo-native interfaces are:

- the storyline notebooks under `notebooks/`, used as the current demo surface
- `xai-demo-data`
- `xai-demo-report` as a secondary legacy report path, still useful for some
  CLI and verification flows but no longer the main presentation layer

## Central components

### Notebook-first demo layer

- `notebooks/overview/`
- `notebooks/shortcut_lab/`
- `notebooks/patchcore_explainability/`
- `notebooks/patchcore_limits/`
- `notebooks/robustness_drift/`
- `notebooks/global_local_explainability/`

Demo 01 is now a real-data Waterbirds notebook with:

- robust repo-root discovery from nested notebook folders
- frozen pretrained ResNet-18 features
- logistic-regression ERM baseline
- group-balanced logistic-regression mitigation
- occlusion and perturbation probes
- exemplar retrieval
- a source-level self-check that rejects reintroduced toy-helper phrasing

Demo 02 is now a real-data NEU industrial shortcut notebook with:

- the prepared `data/processed/neu_cls/shortcut_binary/manifest.jsonl` path as
  the source of truth
- a controlled coloured side-band marker nuisance on real industrial images
- frozen pretrained ResNet-18 features
- logistic-regression ERM baseline
- marker-randomised logistic-regression mitigation
- exact same-image marker counterfactuals
- marker-region ablation, occlusion, and feature-space neighbours
- a source-level self-check that rejects reintroduced toy-helper phrasing and
  weak shortcut/mitigation effects

- report CLI:
  - `src/xai_demo_suite/cli/demo.py`
- data CLI:
  - `src/xai_demo_suite/cli/data.py`
- suite assembly:
  - `src/xai_demo_suite/reports/suite.py`
- shared chrome and cards:
  - `src/xai_demo_suite/reports/report_chrome.py`
  - `src/xai_demo_suite/reports/cards.py`
  - `src/xai_demo_suite/reports/review_pack.py`
- explanation contract and drift helpers:
  - `src/xai_demo_suite/explain/contracts.py`
  - `src/xai_demo_suite/explain/drift.py`
- PatchCore-local evaluation and visual assembly:
  - `src/xai_demo_suite/evaluate/localisation.py`
  - `src/xai_demo_suite/vis/image_panels.py`

## Current local-data shape

- fresh clones can still build the synthetic suite
- local prepared datasets unlock the stronger real-data paths
- MVTec AD bottle is the main local anomaly path for Demo 03
- MVTec LOCO AD, MVTec AD 2, VisA, Waterbirds, MetaShift, NEU-CLS, and KSDD2
  extend specific demos when prepared locally
- optional Torch and Torchvision dependencies are only needed for the serious
  ML-backed paths

## Active concerns

- preserve the explanation contract across demos instead of letting each report
  invent its own semantics
- keep PatchCore provenance explicit and testable
- keep notebooks readable enough to hand to external reviewers while still
  avoiding hidden logic
- keep Demo 01 and Demo 02 on their real-data ResNet plus logistic-regression
  paths rather than drifting back to toy simulators
- avoid notebook drift by moving genuinely reusable logic into package code
- keep local-data assumptions explicit in commands, docs, and final summaries
- keep old report paths secondary and avoid drifting back into notebook wrappers

## Important constraints

- raw datasets must stay out of git
- `data/raw/` is append-only from the repo's point of view
- notebooks are the current demo surface, but important reusable logic should
  still not live only in notebooks unless the task explicitly chooses a
  self-contained notebook artefact
- explanation-supporting images need a verification path or a clearly stated
  caveat

## Ambiguities to remember

- not every report path is equally strong on a fresh clone because several real
  dataset branches are optional
- optional ML-backed PatchCore paths are not the same thing as the deterministic
  fallback path
- generated local reports can look polished while still depending on local data
  availability or non-benchmark settings

## Read first in a new thread

1. `AGENTS.md`
2. `README.md`
3. `docs/codex/ACTIVE_CONTEXT.md`
4. `docs/codex/PROJECT_MAP.md`
5. `docs/XAI_CONTRACT.md`
6. `docs/codex/WORKFLOW.md`
7. `docs/codex/USING_CODEX.md`
8. the relevant task file under `docs/tasks/`
