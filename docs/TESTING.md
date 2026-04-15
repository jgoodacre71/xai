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
At least one notebook per lab should run against a tiny fixture dataset or cached toy artefacts.

## Required quality checks

- Ruff
- MyPy
- Pytest
- coverage on the package code
- notebook smoke on at least the overview notebook and one hero demo

## Testing principle

The point is not only to check correctness. It is to keep Codex grounded by giving it fast feedback loops.
