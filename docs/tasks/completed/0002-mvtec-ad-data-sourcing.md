# 0002-mvtec-ad-data-sourcing: MVTec AD data sourcing workflow

## Status
Complete

## Owner
Codex thread

## Why
The repository starts from scratch and must source MVTec AD without committing
raw data. Demo 03 depends on a reproducible local workflow for fetching,
unpacking, and indexing MVTec AD categories before PatchCore can be implemented.

## Source docs
- REPO_SPEC.md
- docs/DATASETS.md
- docs/PATCHCORE_NOTES.md
- docs/runbooks/add_dataset.md
- docs/decisions/ADR-0003-explanation-contract.md
- Official MVTec AD dataset and download pages

## Scope
- Record official MVTec AD source, licence, restrictions, and category archive
  metadata.
- Add a command-line data workflow for listing, fetching, and preparing MVTec AD
  categories.
- Keep downloaded archives and extracted images out of git.
- Add manifest generation for prepared local data.
- Add tests that use small synthetic archives instead of downloading real data.

## Out of scope
- Downloading the full 4.9 GB dataset during setup.
- PatchCore model training.
- Notebook creation.
- Checksum validation for official archives, unless checksums are published by
  the provider.

## Deliverables
- Data registry updates.
- MVTec AD downloader module.
- Manifest builder.
- CLI entry point.
- Unit tests for metadata, download decisions, extraction safety, and manifests.
- Documentation update with source and usage commands.

## Constraints
- Do not commit raw datasets.
- Never overwrite files in `data/raw/`.
- Preserve source archive structure.
- Keep processing reproducible from committed code.
- Explicitly note the CC BY-NC-SA 4.0 non-commercial restriction.

## Affected files
- `data_registry.yaml`
- `docs/DATASETS.md`
- `pyproject.toml`
- `src/xai_demo_suite/data/`
- `src/xai_demo_suite/cli/`
- `tests/unit/`

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/python -m xai_demo_suite.cli.data list`
5. `./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_ad --category bottle --dry-run`

## Acceptance criteria
- The repo records MVTec AD's official source and licence restrictions.
- A category archive can be fetched into `data/raw/mvtec_ad/archives/` only when
  explicitly requested.
- Existing raw archives are not overwritten unless an explicit overwrite flag is
  provided.
- Local preparation extracts into `data/interim/` and writes a manifest into
  `data/processed/`.
- Tests pass without network access or real MVTec AD files.

## Risks
- MVTec-hosted file links may change; the registry documents the source page so
  links can be refreshed.
- MVTec does not publish checksums on the download page, so integrity validation
  initially relies on archive readability and manifest checks.

## Progress log
### 2026-04-15
- Completed: official MVTec AD source metadata, category archive links, local
  fetch/prepare CLI, manifest generation, docs, and tests.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`,
  `./.venv/bin/pytest -q`, `./.venv/bin/python -m xai_demo_suite.cli.data list`,
  and `./.venv/bin/xai-demo-data fetch mvtec_ad --category bottle --dry-run`
  passed.
- Remaining: download one category locally when ready to start PatchCore
  development.
