# XAI Demo Suite

This repository contains a curated suite of explainable AI demos for vision,
focused on:

- shortcut learning demos;
- PatchCore explainability;
- industrial anomaly detection;
- model limitations such as count, severity, and logic;
- explanation drift under shift.

The working principle is: package code is the product; notebooks are the
showroom. Reusable logic belongs in `src/xai_demo_suite/`, while notebooks
should remain thin narrative and orchestration layers.

## Getting started

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e . pytest ruff mypy pytest-cov
./.venv/bin/ruff check .
./.venv/bin/mypy src
./.venv/bin/pytest -q
```

## Data

Raw datasets are not committed. MVTec AD is sourced from the official MVTec
download page and stored locally under `data/raw/` when explicitly fetched.

```bash
./.venv/bin/xai-demo-data list
./.venv/bin/xai-demo-data fetch mvtec_ad --category bottle --dry-run
./.venv/bin/xai-demo-data fetch mvtec_ad --category bottle
./.venv/bin/xai-demo-data prepare mvtec_ad --category bottle
```

## Optional ML Dependencies

The base package and tests do not require Torch. Install the optional ML stack
when working on the deep PatchCore path:

```bash
./.venv/bin/python -m pip install -e ".[ml]"
```

The current ResNet extractor defaults to random weights and does not download
pretrained weights implicitly.

## First Local Report

After preparing MVTec AD bottle and installing the optional ML stack, generate
the first static PatchCore report slice with:

```bash
./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128
```

The report is written to `outputs/patchcore_bottle/index.html`. Generated
reports and cached model artefacts are ignored by git.

## Main files

- `REPO_SPEC.md` — the long-form repository specification
- `AGENTS.md` — short always-on repo guidance for Codex
- `.agents/PLANS.md` — execution-plan template
- `.agents/skills/` — reusable workflow skills
- `.codex/agents/` — optional specialised subagents
- `docs/` — source-of-truth documentation skeleton
- `data_registry.yaml` — dataset metadata placeholders
- `src/xai_demo_suite/` — reusable package code
- `tests/` — unit and integration tests

Use `docs/tasks/active/` for substantial work so another engineer or Codex
thread can resume from checked-in context.
