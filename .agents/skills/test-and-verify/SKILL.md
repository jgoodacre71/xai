---
name: test-and-verify
description: Run focused validation for a completed task. Use when a change claims to be finished and you want targeted lint, type, test, and smoke verification with a concise report of what passed, what failed, and what remains unverified.
---

# Test and Verify

## Workflow

1. Read `AGENTS.md`.
2. Identify the smallest useful validation commands for the touched files.
3. Run targeted checks first.
4. Escalate to broader checks only if needed.
5. Summarise:
   - commands run;
   - pass / fail status;
   - residual gaps;
   - whether docs and notebook updates were covered.

## Reporting

Be explicit. Do not simply say "looks good".
