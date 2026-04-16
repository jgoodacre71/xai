# ADR-0002: Notebook policy

## Status
Accepted

## Decision
Notebooks are presentation artefacts and thin orchestration layers. Shared
logic must live in package code. Notebook sources are paired with percent-style
text scripts so the narrative layer stays reviewable in git.

## Rationale
Notebook-only logic is hard to test, review, refactor, and reuse.

## Consequences
- more up-front package design;
- cleaner diffs;
- better Codex performance;
- safer long-term maintenance.
- output-free `.ipynb` files in git;
- paired Jupytext-style `.py` notebook sources;
- notebook smoke checks can run against the text sources without hidden logic.
