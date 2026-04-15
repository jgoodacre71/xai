---
name: demo-planner
description: Create or refine a demo plan for this repository. Use when adding a new demo, tightening the narrative of an existing demo, or converting a vague idea into a concrete task packet, notebook outline, and acceptance criteria.
---

# Demo Planner

## Purpose

Turn a fuzzy demo idea into a clean repository task.

## Workflow

1. Read `AGENTS.md`, `REPO_SPEC.md`, `docs/DEMO_CATALOGUE.md`, and `docs/XAI_CONTRACT.md`.
2. Identify whether the demo is mainly about:
   - shortcuts;
   - PatchCore explainability;
   - PatchCore limits;
   - robustness / explanation drift.
3. Propose:
   - title;
   - learning goals;
   - dataset;
   - model(s);
   - explanation artefacts;
   - failure or pitfall;
   - intervention;
   - notebook structure;
   - tests and verification.
4. Write or update a task file rather than leaving the plan only in chat.

## Output format

Return:
- a short summary;
- the suggested task file path;
- a proposed notebook name;
- required source-code modules;
- acceptance criteria;
- risks.
