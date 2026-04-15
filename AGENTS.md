# AGENTS.md

## Purpose

This repository contains a curated suite of XAI demos for vision, centred on:
- shortcut learning and spurious correlations;
- PatchCore and industrial anomaly detection explainability;
- model pitfalls, limits, and explanation drift.

Treat this file as the **table of contents**, not the encyclopedia. The source of truth lives in `docs/`.

## Read this first

Before substantial work, read:
1. `REPO_SPEC.md`
2. `docs/ARCHITECTURE.md`
3. `docs/XAI_CONTRACT.md`
4. the relevant task file under `docs/tasks/`
5. any relevant ADRs in `docs/decisions/`

## Working style

- Use **UK English** in comments, docs, notebook markdown, and user-facing text.
- Prefer clear, typed, well-factored Python over clever shortcuts.
- Keep notebooks for narrative and orchestration only; move reusable logic into `src/`.
- Add or update tests with every meaningful code change.
- Update docs when structure, workflow, or behaviour changes.

## Repo rules

- Do not commit raw datasets.
- Never overwrite files in `data/raw/`.
- All shared logic belongs in `src/xai_demo_suite/`.
- Notebooks must not become the only place where important logic exists.
- PatchCore explanation work must preserve provenance: source image ids and patch coordinates matter.
- Any explanation image used to support a claim should have a corresponding counter-test or verification path.

## Codex workflow rules

- Use **one thread per task**, not one mega-thread for the whole repo.
- For multi-step work, start with a plan using `.agents/PLANS.md`.
- If a correction has been repeated twice, encode it in docs, a skill, or this file.
- Avoid parallel edits to the same files unless isolated with a worktree.
- Prefer the smallest relevant validation step, then report exactly what was run.

## Verification expectations

At minimum, run whichever of these are relevant:
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest -q`
- targeted notebook smoke checks if notebook behaviour changed

If a command cannot be run, state what would be run and why it was skipped.

## Directory guide

- `src/xai_demo_suite/` — package code
- `tests/` — automated tests
- `notebooks/` — narrative demos only
- `docs/` — specs, ADRs, tasks, runbooks
- `.codex/agents/` — optional custom agents
- `.agents/skills/` — reusable Codex skills

## When to use a plan

Use a plan for:
- new demos
- major refactors
- data pipeline additions
- new shared abstractions
- notebook-to-package migrations
- anything likely to touch several files or require design trade-offs

## Done means

A task is done only when:
- code is implemented cleanly;
- relevant tests exist and have been run or explicitly scoped;
- docs/notebooks are updated where needed;
- remaining risks are stated plainly.
