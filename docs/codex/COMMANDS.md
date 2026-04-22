# Commands

## Environment setup

Preferred `uv` path:

```bash
uv sync --group dev
```

Add optional ML dependencies when the task needs Torch-backed paths:

```bash
uv sync --group dev --group ml
```

Plain `venv` fallback:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
```

Add the optional ML stack in the fallback path with:

```bash
./.venv/bin/python -m pip install -e ".[dev,ml]"
```

## Core validation

Preferred default validation:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Plain `venv` fallback:

```bash
./.venv/bin/ruff check .
./.venv/bin/mypy src
./.venv/bin/pytest -q
```

Focused notebook smoke:

```bash
uv run pytest tests/unit/test_notebook_smoke.py -q
```

Focused Demo 01 notebook smoke:

```bash
uv run pytest tests/unit/test_notebook_smoke.py -q -k '01_waterbirds_shortcut'
```

Direct notebook execution without storing outputs in git:

```bash
./.venv/bin/python -m pytest tests/unit/test_notebook_smoke.py -q -k '01_waterbirds_shortcut'
```

Focused CLI smoke:

```bash
uv run pytest tests/integration/test_cli_end_to_end.py -q
```

## Suite generation

Synthetic suite:

```bash
uv run xai-demo-report suite
```

Full local suite when MVTec AD bottle is prepared:

```bash
uv run xai-demo-report suite --include-mvtec
```

Stronger local presentation suite:

```bash
uv run xai-demo-report suite \
  --include-mvtec \
  --mvtec-feature-extractor feature_map_resnet18_pretrained \
  --mvtec-max-train 20 \
  --mvtec-max-examples 3 \
  --mvtec-coreset-size 512 \
  --mvtec-input-size 224
```

Verify generated report structure and semantics:

```bash
uv run xai-demo-report verify
```

## Focused report commands

Hero PatchCore report:

```bash
uv run xai-demo-report patchcore-bottle --max-examples 3
```

Waterbirds shortcut report:

```bash
uv run xai-demo-report waterbirds-shortcut
```

Industrial shortcut report against a prepared alternative manifest:

```bash
uv run xai-demo-report shortcut-industrial \
  --real-manifest-path data/processed/ksdd2/shortcut_binary/manifest.jsonl
```

Explanation drift report against the same alternative industrial manifest:

```bash
uv run xai-demo-report explanation-drift \
  --industrial-manifest-path data/processed/ksdd2/shortcut_binary/manifest.jsonl
```

## Data commands

List supported datasets:

```bash
uv run xai-demo-data list
```

MVTec AD bottle:

```bash
uv run xai-demo-data fetch mvtec_ad --category bottle --dry-run
uv run xai-demo-data fetch mvtec_ad --category bottle
uv run xai-demo-data prepare mvtec_ad --category bottle
```

Waterbirds:

```bash
uv run xai-demo-data fetch waterbirds --category waterbird_complete95_forest2water2 --dry-run
uv run xai-demo-data fetch waterbirds --category waterbird_complete95_forest2water2
uv run xai-demo-data prepare waterbirds --category waterbird_complete95_forest2water2
```

VisA:

```bash
uv run xai-demo-data fetch visa --category all --dry-run
uv run xai-demo-data fetch visa --category all
uv run xai-demo-data prepare visa --category all
```

## Notes and constraints

- `uv` is the preferred command path in this repository.
- The active demo surface is the notebooks, not checked-in HTML reports.
- Some notebook and report paths require prepared local datasets.
- Demo 01 and other ML-backed notebook paths require optional Torch and
  Torchvision dependencies and may rely on locally cached pretrained weights.
- Direct notebook execution through the project `.venv` is more reliable than a
  separate Jupyter environment unless that environment also has the optional ML
  stack installed.
- Generated outputs and caches are local artefacts unless a task explicitly
  asks to curate or publish them.
