# Session Log

## 2026-04-22

- purpose:
  - add a repo-native Codex memory layer for the XAI demo suite so future work
    depends less on thread memory
- files created:
  - `.codex/agents/worker.toml`
  - `docs/codex/PROJECT_MAP.md`
  - `docs/codex/ARCHITECTURE.md`
  - `docs/codex/WORKFLOW.md`
  - `docs/codex/COMMANDS.md`
  - `docs/codex/SAFETY_RULES.md`
  - `docs/codex/ACTIVE_CONTEXT.md`
  - `docs/codex/SESSION_LOG.md`
  - `docs/codex/TASK_TEMPLATE.md`
  - `docs/codex/HANDOFF_TEMPLATE.md`
  - `docs/codex/USING_CODEX.md`
  - `docs/tasks/completed/0047-codex-memory-layer.md`
- files updated:
  - `.codex/config.toml`
- key findings:
  - the repository already had strong repo rules, task history, and specialised
    Codex agents, but it lacked a compact `docs/codex/` memory pack
  - the active surface is the eight-demo report suite plus the shared CLI,
    data, and report infrastructure
  - the main Codex-specific gaps were durable read-first context, command
    summaries, safety reminders, and a session log
- unresolved uncertainties:
  - some stronger report paths still depend on optional local datasets and
    optional ML dependencies, so future task summaries still need to state what
    was actually available
  - the specialised local agent set may need more tuning later, but the primary
    memory gap is now addressed in repo files
- recommended next step:
  - treat `docs/codex/ACTIVE_CONTEXT.md` and this log as part of the normal
    update path whenever commands, active demos, or working assumptions change

## 2026-04-22

- purpose:
  - reorganise the notebook layer by storyline and treat notebooks as the main
    local demo walkthrough surface for now
- files updated:
  - `src/xai_demo_suite/notebooks.py`
  - `notebooks/README.md`
  - `notebooks/overview/00_overview.py`
  - `notebooks/shortcut_lab/01_waterbirds_shortcut.py`
  - `notebooks/shortcut_lab/02_industrial_shortcut_trap.py`
  - `notebooks/patchcore_explainability/03_patchcore_mvtec_ad.py`
  - `notebooks/patchcore_explainability/04_patchcore_wrong_normal.py`
  - `notebooks/patchcore_limits/05_patchcore_count_limit.py`
  - `notebooks/patchcore_limits/06_patchcore_severity_limit.py`
  - `notebooks/patchcore_limits/07_patchcore_loco_logic_limit.py`
  - `notebooks/robustness_drift/08_explanation_drift.py`
  - `notebooks/global_local_explainability/09_global_vs_local_explainability_shap.py`
  - `docs/DEMO_CATALOGUE.md`
  - `docs/DEMO_STATUS.md`
  - `README.md`
  - `tests/unit/test_notebooks.py`
  - `tests/unit/test_notebook_smoke.py`
- key findings:
  - the old flat notebook layout made the repo harder to navigate by storyline
  - the existing notebooks already had the right narrative scaffold, but they
    were too dependent on external HTML viewing
  - rendering the generated HTML content inline gives a much better notebook
    walkthrough without changing the report builders themselves
- unresolved uncertainties:
  - the SHAP storyline notebook is still more prototype-like than the main
    report-backed demos because it keeps more logic inside the notebook
  - future work should still move more reusable notebook logic into `src/`
- recommended next step:
  - keep polishing the notebooks as the main local demo surface, then decide
    later whether a separate presentation layer should sit on top of them

## 2026-04-22

- purpose:
  - convert the repo decisively toward notebook-first demos, rebuild Demo 01 as
    a real-data Waterbirds notebook, and tighten the repo-native Codex memory
    so future threads start from the new working surface
- files updated:
  - `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb`
  - `tests/unit/test_notebooks.py`
  - `tests/unit/test_notebook_smoke.py`
  - `docs/tasks/active/0053-notebook-native-demos.md`
  - `docs/codex/ACTIVE_CONTEXT.md`
  - `docs/codex/PROJECT_MAP.md`
  - `docs/codex/WORKFLOW.md`
  - `docs/codex/COMMANDS.md`
- key findings:
  - the old HTML/report-wrapper model was no longer acceptable for the current
    phase; the notebooks are now the demo artefacts
  - Demo 01 is strongest when it stays real-data only, uses frozen pretrained
    ResNet-18 features plus logistic-regression heads, and makes the shortcut
    legible through group metrics, occlusion, perturbation, and exemplar
    retrieval
  - notebook execution is validated cleanly through the project `.venv`, while
    a separate Jupyter environment may still fail if it lacks `torchvision`
  - the repo still enforces output-free notebooks in git, so execution should
    be followed by stripping stored outputs unless repo policy changes later
- unresolved uncertainties:
  - some remaining notebooks still need the same level of visual and narrative
    polish as Demo 01
  - the old report builders remain in the tree and could confuse future work if
    the notebook-first surface is not stated explicitly in memory and docs
- recommended next step:
  - treat `docs/codex/ACTIVE_CONTEXT.md` and this log as living records of the
    notebook-first surface and update them whenever a demo notebook becomes the
    practical source of truth
