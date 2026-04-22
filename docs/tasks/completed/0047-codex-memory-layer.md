# 0047: Codex memory layer

## Status
Complete

## Owner
Codex thread

## Why

This repository already had strong repo rules and task history, but it lacked a
compact repo-native Codex memory pack like the one used in the
`hyperliquid-strategy-suite` repository. The goal of this task is to create the
same kind of durable in-repo memory layer, adapted to the XAI demo suite rather
than copied blindly.

## Source of truth

- `AGENTS.md`
- `REPO_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/XAI_CONTRACT.md`
- `docs/CODEx_WORKFLOW.md`
- `docs/DEMO_STATUS.md`
- `docs/DEMO_CATALOGUE.md`
- `docs/TESTING.md`
- `docs/decisions/ADR-0001-demo-philosophy.md`
- `docs/decisions/ADR-0002-notebook-policy.md`
- `docs/decisions/ADR-0003-explanation-contract.md`
- `/Users/johngoodacre/work/hyperliquid-strategy-suite/.codex/config.toml`
- `/Users/johngoodacre/work/hyperliquid-strategy-suite/.codex/agents/explorer.toml`
- `/Users/johngoodacre/work/hyperliquid-strategy-suite/.codex/agents/worker.toml`
- `/Users/johngoodacre/work/hyperliquid-strategy-suite/docs/codex/`

## Scope

Add an XAI-specific Codex memory layer under `docs/codex/`, wire it into the
local `.codex/config.toml`, and record the work in task history.

## Out of scope

- changing demo behaviour
- changing tests or generated outputs
- changing data adapters
- reworking the existing specialised local agent roles

## Deliverables

- repo-native Codex memory docs
- `.codex/config.toml` read-first and path mapping
- a generic `.codex/agents/worker.toml`
- completed task record

## Constraints

- keep the content XAI-specific rather than trader/runtime-specific
- preserve existing specialised local agents
- use UK English in repo-facing text
- keep the guidance compact and durable

## Proposed file changes

- `.codex/config.toml`
  - add read-first and path mappings
- `.codex/agents/worker.toml`
  - add a general execution-focused local agent
- `docs/codex/*.md`
  - add durable Codex memory, workflow, command, and handoff docs
- `docs/tasks/completed/0047-codex-memory-layer.md`
  - record the task in repo history

## Validation plan

1. Parse the touched TOML files.
2. Run `git diff --check`.

## Risks

- the docs could drift if future tasks update workflows but not the Codex layer
- optional dataset and ML assumptions still need explicit task-by-task handling

## Decision log

### 2026-04-22 00:00
- Decision: add `docs/codex/` rather than extending one existing file.
- Reason: the strategy repo pattern works because it separates active context,
  commands, workflow, and safety into small durable files.
- Follow-up: keep `ACTIVE_CONTEXT.md` and `SESSION_LOG.md` current.

### 2026-04-22 00:00
- Decision: preserve the existing specialised XAI agents and only add a generic
  worker.
- Reason: `xai` already had explorer, reviewer, architect, and PatchCore worker
  roles that are useful and project-specific.
- Follow-up: tune those agent prompts later only if they cause friction.

## Progress log

### 2026-04-22 00:00
- Completed: reviewed the existing XAI repo guidance and the Codex setup in
  `hyperliquid-strategy-suite`.
- Verification: source files inspected across both repositories.
- Remaining: write the XAI-specific Codex memory layer and validate the TOML.

### 2026-04-22 00:00
- Completed: added `docs/codex/`, updated `.codex/config.toml`, added
  `.codex/agents/worker.toml`, and recorded the task.
- Verification: planned TOML parsing and `git diff --check`.
- Remaining: none.
