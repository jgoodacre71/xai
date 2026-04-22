# Using Codex In This Repository

## Default expectation

Codex should use the repo-native Codex layer as durable working context for
this repository.

That means:

- read the Codex docs first
- inspect before editing for non-trivial tasks
- keep reusable logic in `src/xai_demo_suite/`
- preserve the explanation contract and PatchCore provenance
- state when a task depends on optional local data or ML dependencies

## How to start a task

Good requests usually include:

- goal
- scope
- constraints
- whether you want analysis only or implementation
- the relevant demo, dataset, or report path if known

Examples:

- "Review Demo 03 provenance logic. No edits yet."
- "Fix the Waterbirds report and update the matching tests."
- "Investigate the suite verifier without changing notebook content."
- "Update the local Codex docs after this workflow change."

## When to say "no edits yet"

Say this when you want:

- repo mapping
- architecture tracing
- impact analysis
- review before implementation
- file or test identification

That should bias the work toward an explorer-style response.

## When to ask for implementation

Ask directly when you want:

- a focused fix
- a controlled refactor
- docs or notebook updates
- tests or validation
- a clean handoff

That should bias the work toward a worker-style response.

## What Codex should do automatically

Codex should not need repeated reminders to:

- read repo-native guidance before editing
- keep notebook logic thin
- validate with the smallest relevant command set
- update docs when workflow or behaviour changes materially
- call out optional-data assumptions instead of hiding them

## Good end-of-task prompts

Useful close-out prompts include:

- "Summarise changes, files, tests, and risks."
- "Give me a handoff."
- "Update the Codex docs if this changed the active workflow."
- "Review only."

## If a new thread is started later

Future threads should begin by reading:

1. `AGENTS.md`
2. `README.md`
3. `docs/codex/ACTIVE_CONTEXT.md`
4. `docs/codex/PROJECT_MAP.md`
5. `docs/XAI_CONTRACT.md`
6. `docs/codex/WORKFLOW.md`
7. `docs/codex/USING_CODEX.md`
8. `docs/codex/SAFETY_RULES.md`

Then inspect only the files needed for the requested task.
