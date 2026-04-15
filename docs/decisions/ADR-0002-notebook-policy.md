# ADR-0002: Notebook policy

## Status
Accepted

## Decision
Notebooks are presentation artefacts and thin orchestration layers. Shared logic must live in package code.

## Rationale
Notebook-only logic is hard to test, review, refactor, and reuse.

## Consequences
- more up-front package design;
- cleaner diffs;
- better Codex performance;
- safer long-term maintenance.
