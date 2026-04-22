# 0053: Notebook-native demo migration

## Status
In progress

## Owner
Codex thread

## Why
The current active notebooks are still thin wrappers around HTML report builders.
That is the wrong product surface for the current phase of this repository.
For now, the notebooks are the demos, so each demo notebook needs to contain the
story, runnable code, and visible graphics directly in the notebook itself.

## Source of truth
- REPO_SPEC.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- AGENTS.md
- docs/tasks/completed/0050-storyline-notebook-refactor.md
- docs/tasks/completed/0051-notebook-surface-simplification.md
- docs/tasks/completed/0052-notebook-only-surface-and-execution-pass.md

## Scope
- Replace report-builder notebook cells in demos 01 to 08.
- Make each active non-SHAP notebook run as a notebook-native demo.
- Keep the markdown story and place figures near the interpreting text.
- Prefer small synthetic or lightweight local flows where they make the demo
  clearer and easier to run.
- Retain only low-level shared helpers in `src/` where reuse is genuinely useful.

## Out of scope
- Rebuilding the SHAP notebook in this task.
- Final presentation polish or slide export.
- Deleting the HTML reporting stack from the repository entirely.
- Reworking every report module in `src/xai_demo_suite/reports/`.

## Deliverables
- rewritten notebooks under `notebooks/` for demos 01 to 08;
- updated tests for notebook-native execution;
- updated docs that describe notebooks as the live demo surface;
- removal or de-emphasis of notebook helpers that assume HTML report outputs.

## Constraints
- Use UK English in notebook markdown and docs.
- Do not rely on generated HTML pages as the active demo surface.
- Do not make the notebooks opaque wrappers around hidden logic.
- Keep the demos runnable in a local notebook session with visible figures.
- Respect the XAI contract where appropriate: evidence, provenance,
  counterfactual change, and stability.

## Proposed file changes
- `notebooks/shortcut_lab/*.ipynb`: notebook-native shortcut demos.
- `notebooks/patchcore_explainability/*.ipynb`: notebook-native PatchCore demos.
- `notebooks/patchcore_limits/*.ipynb`: notebook-native limitation demos.
- `notebooks/robustness_drift/*.ipynb`: notebook-native drift demo.
- `tests/unit/test_notebooks.py`: validate notebook-native structure, not report imports.
- `tests/unit/test_notebook_smoke.py`: execute notebooks without expecting HTML outputs.
- `README.md`, `notebooks/README.md`, `docs/DEMO_CATALOGUE.md`,
  `docs/DEMO_STATUS.md`, `docs/TESTING.md`: align docs with notebook-native demos.
- `src/xai_demo_suite/notebooks.py`: remove or reduce report-oriented notebook helpers.

## Validation plan
1. `./.venv/bin/python -m compileall src tests`
2. `./.venv/bin/pytest tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py -q`
3. targeted notebook smoke execution for demos 01 to 08
4. `./.venv/bin/ruff check src tests`
5. `git diff --check`

## Risks
- Some existing demos are tightly coupled to HTML-report code and may need
  simplified notebook-native versions first.
- Optional dependencies such as `torch` or `torchvision` may make certain paths
  less portable; fallback synthetic demonstrations may be needed.
- There is a tension between fully self-contained notebooks and the repository
  rule that shared logic should live in `src/`; this task should keep
  notebook-visible demo logic in the notebook while leaving reusable primitives
  in package code where sensible.

## Decision log
### 2026-04-22 12:04
- Decision: treat HTML reports as secondary legacy surfaces, not the active demo interface.
- Reason: the user wants the notebook itself to be the demo for now.
- Follow-up: rewrite the notebooks rather than further polishing report wrappers.

### 2026-04-22 12:42
- Decision: active non-SHAP demo notebooks should not depend on `matplotlib`.
- Reason: the base project environment does not ship with `matplotlib`, and the
  demos need to be portable standalone artefacts for local walkthroughs and
  external model review.
- Follow-up: use PIL-based panels and notebook-native image display instead.

## Progress log
### 2026-04-22 12:04
- Completed: wrote the active task plan and confirmed the current notebooks still import report builders.
- Verification: inspected notebooks and current tests.
- Remaining: rewrite demos 01 to 08, update tests/docs, and run smoke checks.

### 2026-04-22 12:42
- Completed: rewrote demos 01 to 08 as notebook-native walkthroughs, removed
  the obsolete notebook HTML helper, deleted the checked-in `outputs/`
  artefacts, and updated notebook tests to enforce the new contract.
- Verification: manual smoke execution of notebooks `00` to `08`,
  `./.venv/bin/pytest tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py -q`,
  `./.venv/bin/ruff check tests`, `./.venv/bin/python -m compileall src tests`,
  and `git diff --check`.
- Remaining: align any remaining legacy report-first docs if the report stack
  is kept around as a secondary surface.

### 2026-04-22 18:05
- Completed: rebuilt Demo 01 as a self-contained, real-data-first Waterbirds
  notebook with explicit dataset mode, manifest-backed group counts, a learned
  ERM baseline, a centre-crop mitigation model, notebook-native PIL figures,
  occlusion sensitivity, an approximate real-data habitat counterfactual, and
  final self-check assertions.
- Verification: direct top-to-bottom execution of
  `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb`,
  `./.venv/bin/pytest tests/unit/test_notebooks.py -q`, and
  `git diff --check`.
- Remaining: apply the same standard of rewrite and visual polish to the other
  active demo notebooks.

### 2026-04-22 21:10
- Decision: Demo 01 must be real-data only and must not silently fall back to a
  synthetic cartoon path.
- Reason: the active requirement is now a credible Waterbirds notebook that can
  be handed to external reviewers and improved in-place, not a toy explainer.
- Follow-up: remove any synthetic helper path from Demo 01 and fail fast when
  the Waterbirds manifest is missing.

### 2026-04-22 21:18
- Decision: allow `matplotlib` in notebook-native demos when it materially
  improves figure quality and readability.
- Reason: the old blanket prohibition forced rough PIL-only charts that were
  not good enough for the notebook-first demo surface.
- Follow-up: update notebook validation so it blocks hidden report imports and
  stale output paths, but no longer blocks `matplotlib`.

### 2026-04-22 21:36
- Completed: replaced Demo 01 with a real-data Waterbirds notebook built around
  a frozen pretrained ResNet-18 feature extractor, logistic-regression ERM and
  group-balanced heads, real-image group grids, manifest heatmaps, occlusion
  sensitivity, real-data perturbation probes, nearest-exemplar retrieval, and
  before/after mitigation re-test panels.
- Verification: direct top-to-bottom execution of
  `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb`.
- Remaining: run the targeted notebook tests and smoke checks after updating
  the validation contract.

### 2026-04-22 22:12
- Completed: hardened Demo 01 for notebook use from nested working
  directories, removed any reliance on online weight downloads by loading the
  cached local `resnet18-f37072fd.pth` checkpoint directly, and tightened the
  notebook contract around real-data-only execution and the ResNet-based model.
- Verification: direct top-to-bottom execution of
  `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb` from both the repo root
  and `notebooks/shortcut_lab/`, `./.venv/bin/pytest tests/unit/test_notebooks.py -q`,
  `./.venv/bin/pytest tests/unit/test_notebook_smoke.py -q -k '01_waterbirds_shortcut'`,
  and `git diff --check`.
- Remaining: carry the same standard of execution robustness and visual polish
  into the remaining active notebooks.

### 2026-04-22 22:54
- Completed: polished Demo 01 around the working real-data ResNet path rather
  than rebuilding it again. Added a stale-kernel guard, seeded group sampling,
  representative sample overrides for the explanatory grid, explanation-aware
  failure selection, manual 224x224 model-canvas alignment, feature caching
  under `data/artefacts/notebooks/01_waterbirds_resnet18_features.npz`, more
  interpretable occlusion panels, a cleaner exemplar grid, stronger mitigation
  summary badges, and more careful wording for mixed local evidence.
- Verification: direct top-to-bottom execution of
  `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb` from both the repo root
  and `notebooks/shortcut_lab/`, `./.venv/bin/pytest tests/unit/test_notebooks.py -q`,
  `./.venv/bin/pytest tests/unit/test_notebook_smoke.py -q -k '01_waterbirds_shortcut'`,
  and `git diff --check`.
- Remaining: decide whether to keep the current explanation-optimised failure
  case or replace it with a manually curated override once the visual review is
  complete in Jupyter.

### 2026-04-22 23:32
- Completed: finished the final polish pass for Demo 01. Tightened the
  top-level promise, made the Waterbirds compositing note explicit, upgraded
  the metric dashboard badges and wrapped x-axis labels, changed the selected
  failure labelling to `strong background/context sensitivity` when the
  perturbation and occlusion evidence support it, added a transparent
  "Why this case?" card, improved the occlusion and real-data probe figures,
  redesigned exemplar retrieval into cleaner labelled rows, made mitigation
  limits explicit, strengthened the same-case verdict, and extended the final
  audit card and self-check so the notebook source itself rejects any
  reintroduced toy-helper phrasing.
- Verification: direct top-to-bottom execution of
  `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb` from the repo root and
  from `notebooks/shortcut_lab/`, `./.venv/bin/python -m pytest
  tests/unit/test_notebooks.py -q`, `./.venv/bin/python -m pytest
  tests/unit/test_notebook_smoke.py -q -k '01_waterbirds_shortcut'`, and
  `git diff --check`.
- Remaining: `jupyter nbconvert --execute` still depends on an environment with
  both `nbconvert` and `torchvision` available. The project `.venv` execution
  path is validated; the separate `qst` Jupyter environment still fails because
  `torchvision` is missing there.

### 2026-04-22 23:58
- Completed: applied the final visual and narrative polish pass to Demo 01
  without changing its model or data contract. Replaced the raw run-summary
  dictionary with a polished metric table, tightened the dashboard card layout,
  curated cleaner opening-grid examples through the representative sample
  overrides, reframed the selected failure around background/context evidence,
  added a stronger occlusion callout, improved the real-data probe labelling,
  clarified the frozen-feature wording in the exemplar section, strengthened
  the same-case verdict, differentiated mitigation-improvement cards from the
  remaining-risk card, and fixed the final audit-card wrapping.
- Verification: direct top-to-bottom execution of
  `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb` from the repo root and
  from `notebooks/shortcut_lab/`, `./.venv/bin/python -m pytest
  tests/unit/test_notebooks.py -q`, `./.venv/bin/python -m pytest
  tests/unit/test_notebook_smoke.py -q -k '01_waterbirds_shortcut'`, and
  `git diff --check`.
- Remaining: visual review inside a live Jupyter front end is still the best
  way to judge whether any figure wants one more spacing tweak, but the notebook
  content and execution contract are now stable again.

### 2026-04-23 00:44
- Completed: rebuilt Demo 02 into a real-data NEU industrial shortcut notebook
  using the prepared `shortcut_binary` manifest, frozen pretrained ResNet-18
  features, a logistic-regression ERM baseline, a stamp-randomised mitigation,
  exact same-image stamp counterfactuals, stamp-region ablation, occlusion, and
  exemplar retrieval. Also tightened the notebook contract around robust
  repo-root discovery, feature caching, real split semantics, and source-level
  rejection of the old toy-helper phrases.
- Verification: `./.venv/bin/python -m pytest tests/unit/test_notebooks.py -q`,
  `./.venv/bin/python -m pytest tests/unit/test_notebook_smoke.py -q -k
  '02_industrial_shortcut_trap'`, direct top-to-bottom execution of
  `notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb` from the repo root
  and from `notebooks/shortcut_lab/`, and `git diff --check`.
- Remaining: live front-end visual review is still useful for fine spacing
  polish, but the notebook now executes cleanly and tells the intended shortcut
  story on the real NEU data.

### 2026-04-23 02:07
- Completed: applied the final visual and narrative polish pass to Demo 02
  without changing its real-data architecture. Renamed the nuisance honestly as
  a coloured side-band marker, added defect-region boxes and zoom crops to the
  opening task figure, tightened selected-case choice and caveats, promoted the
  exact marker swap into the hero figure, integrated ablation scores into the
  visual panels, clarified the occlusion and exemplar interpretation, and made
  the mitigation and same-case re-test conclusions more explicit.
- Verification: `./.venv/bin/python -m pytest tests/unit/test_notebooks.py -q`,
  `./.venv/bin/python -m pytest tests/unit/test_notebook_smoke.py -q -k
  '02_industrial_shortcut_trap'`, direct top-to-bottom execution of
  `notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb` from the repo root
  and from `notebooks/shortcut_lab/`, and `git diff --check`.
- Remaining: the notebook is now stable for repo use; any further changes would
  be presentation-level curation rather than structural or execution fixes.
