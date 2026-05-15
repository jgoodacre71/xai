# Data replication workflow

Use this workflow when recreating the project on a new machine or at work.

## 1. Prepare the environment

```bash
uv sync --group dev --group notebooks
```

For deep feature notebooks, also install the ML group:

```bash
uv sync --group dev --group notebooks --group ml
```

Plain virtual environments can use the equivalent editable install:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev,notebooks,ml]"
```

## 2. Check permissions before fetching data

Read:

- `docs/DATA_PERMISSION_MATRIX.md`
- `docs/DATA_REQUIREMENTS.md`
- `data_registry.yaml`

Do not fetch or copy a dataset into a work environment until its row has a
clear work-use decision.

## 3. Audit the local data state

```bash
uv run python scripts/audit_data_inventory.py --root . --output docs/local_data_inventory_local.txt
```

The generated inventory is local evidence only. It should not be treated as a
licence approval.

## 4. Fetch and prepare approved datasets

Use the dataset CLI where supported:

```bash
uv run xai-demo-data list
uv run xai-demo-data fetch waterbirds --category waterbird_complete95_forest2water2 --dry-run
uv run xai-demo-data prepare waterbirds --category waterbird_complete95_forest2water2
```

For datasets with manual archives, place the archive under the documented
`data/raw/<dataset>/archives/` path, or pass an explicit path to the prepare
command. Never overwrite files in `data/raw/`.

## 5. Run generated demos first

Demo 00 is the no-permission opener:

```bash
uv run jupyter nbconvert --execute notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb --to notebook --inplace
```

It should run without any external data.

## 6. Run real-data notebooks only after manifests exist

Real-data notebooks should fail clearly when required manifests are missing.
For example, Demo 01 expects:

```text
data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl
```

Run the inventory script first if a notebook reports missing files.

## 7. Validate before sharing

```bash
uv run pytest tests/unit/test_notebooks.py -q
uv run pytest tests/unit/test_notebook_smoke.py -q
uv run ruff check .
uv run mypy src
```

If any command cannot be run in the target environment, record the exact reason
and the closest equivalent command that was run.
