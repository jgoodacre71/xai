# Codex execution plans for this repository

Use an execution plan for any task that is:
- multi-file;
- architecturally important;
- ambiguous;
- likely to take more than one implementation pass;
- likely to be resumed in a later session.

## Plan file location

Create active plans under:

```text
docs/tasks/active/<task-id>.md
```

Move completed plans to:

```text
docs/tasks/completed/<task-id>.md
```

## Plan template

```markdown
# <task-id>: <short title>

## Status
Planned | In progress | Blocked | Complete

## Owner
<name or Codex thread>

## Why
Why this task exists and why it matters.

## Source of truth
- REPO_SPEC.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/decisions/ADR-xxxx-...
- any dataset or model notes

## Scope
What this task will do.

## Out of scope
What this task will deliberately not do.

## Deliverables
- code files
- tests
- docs
- notebooks
- reports

## Constraints
- architectural
- data
- testing
- UX
- explanation fidelity

## Proposed file changes
List likely files and why each one changes.

## Validation plan
Exact commands to run, in order of smallest useful checks first.

## Risks
What might go wrong.

## Decision log
### <date/time>
- Decision:
- Reason:
- Follow-up:

## Progress log
### <date/time>
- Completed:
- Verification:
- Remaining:
```

## Plan discipline

- Update the plan when the design changes materially.
- Do not rely on conversation history as the only memory.
- Keep the plan concrete enough that another engineer or another Codex thread can resume it.
