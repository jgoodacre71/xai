# Testing

## Test layers

### Unit tests
For:
- pure utility functions
- schema validation
- small image or patch transforms
- metrics and drift calculations

### Integration tests
For:
- dataset adapters
- PatchCore memory-bank metadata handling
- explanation artefact generation
- report generation

### Notebook smoke tests
Notebook smoke runs should execute the paired percent-script sources against
tiny fixture settings or synthetic fallbacks.

## Required quality checks

- Ruff
- MyPy
- Pytest
- coverage on the package code
- notebook smoke on representative paired notebook scripts across the main demo pillars

## Testing principle

The point is not only to check correctness. It is to keep Codex grounded by giving it fast feedback loops.
