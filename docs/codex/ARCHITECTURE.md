# Architecture

## Runtime model

This repository is not a long-running service. The main runtime is a static
artefact pipeline:

1. prepare or discover local dataset manifests;
2. run a report builder through `xai-demo-report`;
3. assemble figures, cards, and HTML;
4. verify the generated output structure.

The product is the package code plus the generated demo artefacts. Notebooks are
the narrative front-end to that package code.

## Main control flow

### Data path

- `src/xai_demo_suite/cli/data.py`
  - CLI dispatch for listing, fetch, and prepare commands
- `src/xai_demo_suite/data/manifests.py`
  - shared dataset registration and preparation helpers
- dataset-specific manifest modules
  - Waterbirds, industrial shortcut, MVTec-family, MetaShift, VisA, and related
    local adapters

The data flow must preserve raw data and write derived material into reproducible
processed or artefact locations.

### Report path

- `src/xai_demo_suite/cli/demo.py`
  - CLI dispatch for per-demo reports, suite generation, verification, and
    review-pack assembly
- `src/xai_demo_suite/reports/*.py`
  - demo-specific report builders
- `src/xai_demo_suite/reports/suite.py`
  - cross-demo index and local presentation hub
- `src/xai_demo_suite/reports/build_metadata.py`
  - build metadata for generated artefacts
- `src/xai_demo_suite/reports/cards.py`
  - demo cards and summary surfaces
- `src/xai_demo_suite/reports/report_chrome.py`
  - shared flagship presentation framing

### Notebook path

- `notebooks/**/*.ipynb`
  - output-free checked-in notebooks organised by storyline

Notebook execution should exercise package code and write local outputs under
the configured notebook smoke output root.

## Shared design rules

- shared logic belongs in `src/xai_demo_suite/`
- notebook-only logic is architectural drift
- explanation artefacts should be traceable back to package code and tests
- PatchCore work must preserve provenance, including source image identity and
  patch coordinates
- any image used to support a claim should have a counter-test or verification
  path

## Validation shape

The repo has a clear default validation ladder:

1. `uv run ruff check .`
2. `uv run mypy src`
3. `uv run pytest -q`
4. focused report or notebook smoke when behaviour changed

When optional ML or local datasets are required, the task should state that
explicitly rather than quietly assuming they exist.
