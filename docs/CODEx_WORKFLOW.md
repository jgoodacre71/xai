# Codex workflow

## Goals

- keep long projects coherent;
- avoid context rot;
- avoid repeating the same instructions;
- keep repository truth in files, not only in chat history.

## Rules

### 1. Start from docs
Read `AGENTS.md` and the relevant docs before changing code.

### 2. Use a plan for difficult work
Anything multi-step should start with a plan in `docs/tasks/active/`.

### 3. Use one thread per task
Do not let one thread become the entire memory of the project.

### 4. Use worktrees for parallel changes
If two active tasks may touch overlapping files, isolate them.

### 5. Promote repeatable prompts into skills
If the same pattern works twice, codify it.

### 6. Update the guidance when friction repeats
If Codex makes the same mistake twice, update docs or skills.

## Good prompt template

Goal:
Context:
Constraints:
Done when:

Always include file references.
