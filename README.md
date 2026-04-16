# XAI Demo Suite

This repository contains a curated suite of explainable AI demos for vision,
focused on:

- shortcut learning demos;
- PatchCore explainability;
- industrial anomaly detection;
- model limitations such as count, severity, and logic;
- explanation drift under shift.

The working principle is: package code is the product; notebooks are the
showroom. Reusable logic belongs in `src/xai_demo_suite/`, while notebooks
should remain thin narrative and orchestration layers.

## Getting started

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e . pytest ruff mypy pytest-cov
./.venv/bin/ruff check .
./.venv/bin/mypy src
./.venv/bin/pytest -q
```

## Build and Verify the Demo Suite

The synthetic demos do not require external data:

```bash
./.venv/bin/xai-demo-report suite
./.venv/bin/xai-demo-report verify
```

On this machine, after preparing MVTec AD bottle, include the hero MVTec report:

```bash
./.venv/bin/xai-demo-report suite --include-mvtec
./.venv/bin/xai-demo-report verify
```

For the stronger local presentation run, build Demo 03 with explicit
ImageNet-pretrained ResNet-18 feature-map PatchCore and a 512-patch coreset:

```bash
./.venv/bin/xai-demo-report suite \
  --include-mvtec \
  --mvtec-feature-extractor feature_map_resnet18_pretrained \
  --mvtec-max-train 20 \
  --mvtec-max-examples 3 \
  --mvtec-coreset-size 512 \
  --mvtec-input-size 224
./.venv/bin/xai-demo-report verify
```

The local presentation index is written to `outputs/index.html`.

## Data

Raw datasets are not committed. MVTec AD is sourced from the official MVTec
download page and stored locally under `data/raw/` when explicitly fetched.

```bash
./.venv/bin/xai-demo-data list
./.venv/bin/xai-demo-data fetch mvtec_ad --category bottle --dry-run
./.venv/bin/xai-demo-data fetch mvtec_ad --category bottle
./.venv/bin/xai-demo-data prepare mvtec_ad --category bottle
```

Waterbirds now has the same explicit local data flow:

```bash
./.venv/bin/xai-demo-data fetch waterbirds --category waterbird_complete95_forest2water2 --dry-run
./.venv/bin/xai-demo-data fetch waterbirds --category waterbird_complete95_forest2water2
./.venv/bin/xai-demo-data prepare waterbirds --category waterbird_complete95_forest2water2
```

When the prepared Waterbirds manifest exists, Demo 01 switches into a real-data
path with frozen ResNet-18 linear probes, worst-group metrics, Grad-CAM,
Integrated Gradients, and simple context-masking perturbation checks. The
synthetic proxy remains as the fallback for fresh clones without local data.

MVTec AD 2 now has a second-wave local adapter:

```bash
./.venv/bin/xai-demo-data fetch mvtec_ad_2 --category all --dry-run
./.venv/bin/xai-demo-data fetch mvtec_ad_2 --category all --archive-url <direct-archive-url>
./.venv/bin/xai-demo-data prepare mvtec_ad_2 --category all
```

The fetch path is intentionally conservative: the official source page is
recorded, but the repo does not hard-code a brittle direct dataset link. If you
already have the archive locally, place it under `data/raw/mvtec_ad_2/archives/`
or pass `--archive-path` to `prepare`.

VisA now has the same local fetch and prepare path:

```bash
./.venv/bin/xai-demo-data fetch visa --category all --dry-run
./.venv/bin/xai-demo-data fetch visa --category all
./.venv/bin/xai-demo-data prepare visa --category all
```

The VisA adapter fetches the published archive plus the upstream one-class split
CSV, then writes one canonical manifest per prepared category under
`data/processed/visa/`.

MetaShift now has a local adapter for the published cat-vs-dog indoor/outdoor
subpopulation-shift split:

```bash
./.venv/bin/xai-demo-data fetch metashift --category subpopulation_shift_cat_dog_indoor_outdoor --dry-run
./.venv/bin/xai-demo-data prepare metashift --category subpopulation_shift_cat_dog_indoor_outdoor
```

This path is intentionally manual about upstream dependencies: generate the
split with the published MetaShift scripts and base assets, place it under
`data/external/metashift/MetaShift-subpopulation-shift/`, then build the local
manifest.

## Optional ML Dependencies

The base package and tests do not require Torch. Install the optional ML stack
when working on the deep PatchCore path:

```bash
./.venv/bin/python -m pip install -e ".[ml]"
```

The local PatchCore report defaults to deterministic colour/texture patch
features, so it does not require Torch. The serious deep-feature path is
available explicitly through a dense ResNet-18 feature-map extractor. It only
uses pretrained weights when `feature_map_resnet18_pretrained` is requested.

## First Local Report

After preparing MVTec AD bottle, generate the first static PatchCore report
slice with:

```bash
./.venv/bin/xai-demo-report patchcore-bottle --max-examples 3
```

The report is written to `outputs/patchcore_bottle/index.html`. When prepared
MVTec masks are available, each selected anomaly also includes a ground-truth
localisation check for the top scored patch. The report also includes local
test-split diagnostics: image-level ROC AUC from the max patch score, defect
type score summaries, and top-patch mask-hit checks. Generated reports, demo
cards, local index files, and cached model artefacts are ignored by git.

After preparing Waterbirds, generate Demo 01 with:

```bash
./.venv/bin/xai-demo-report waterbirds-shortcut
```

You can force the fallback path with `--no-real-data`, or use random backbone
weights for quick local smoke tests with `--weights none`.

You can switch extractors explicitly:

```bash
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor mean_rgb
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor resnet18_random
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_random --coreset-size 512
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_pretrained --coreset-size 512
```

The pretrained command may download Torchvision ResNet-18 weights into the local
Torch cache. That cache, generated reports, and memory-bank artefacts are not
tracked by git.

Use `--benchmark-limit` for quick smoke runs; omit it to score the full local
MVTec AD bottle test split in the report diagnostics.

The narrative notebook for the same demo is checked in at
`notebooks/03_patchcore_mvtec_bottle.ipynb`. It is intentionally output-free and
delegates the implementation to package code.

## PatchCore Limits Report

Generate the synthetic limits demo with:

```bash
./.venv/bin/xai-demo-report patchcore-limits
```

The report is written to `outputs/patchcore_limits/index.html`. It shows slot
board examples where PatchCore-style novelty is useful, but count, severity, and
semantic logic require extra modelling layers.

Generate the real logical-anomaly report with:

```bash
./.venv/bin/xai-demo-report patchcore-logic
```

When the prepared MVTec LOCO AD `juice_bottle` manifest exists, the report at
`outputs/patchcore_logic/index.html` contrasts PatchCore patch novelty with a
category-specific front-label template comparator. That comparator is
deliberately narrow: it is a packaging-rule check for this aligned category,
not a general anomaly model.

## PatchCore Wrong-Normal Report

Generate the synthetic normal-set contamination demo with:

```bash
./.venv/bin/xai-demo-report patchcore-wrong-normal
```

The report is written to `outputs/patchcore_wrong_normal/index.html`. It
compares a clean memory bank with a memory bank contaminated by a corner
acquisition tab.

## Industrial Shortcut Report

Generate the synthetic shortcut demo with:

```bash
./.venv/bin/xai-demo-report shortcut-industrial
```

The report is written to `outputs/shortcut_industrial/index.html`. It shows a
learned corner-stamp shortcut, a stamp-invariant intervention model,
Grad-CAM and Integrated Gradients overlays, and known-region perturbation
diagnostics over the stamp and the part.

## Explanation Drift Report

Generate the synthetic drift demo with:

```bash
./.venv/bin/xai-demo-report explanation-drift
```

The report is written to `outputs/explanation_drift/index.html`. It separates
performance drift from explanation drift for the learned industrial shortcut
models under blur, contrast, compression, lighting, and shadow shifts. When
local MVTec bottle data is prepared, it adds a PatchCore anomaly-drift section
with image-level AUC, top-patch movement, and mask-coverage checks. When local
MVTec AD 2 scenario manifests are prepared, the same report now adds second-wave
anomaly-drift sections for those scenarios as well. When local VisA manifests
are prepared, the same report adds cross-dataset anomaly-drift sections there
too.

## Main files

- `REPO_SPEC.md` — the long-form repository specification
- `AGENTS.md` — short always-on repo guidance for Codex
- `.agents/PLANS.md` — execution-plan template
- `.agents/skills/` — reusable workflow skills
- `.codex/agents/` — optional specialised subagents
- `docs/` — source-of-truth documentation skeleton
- `data_registry.yaml` — dataset metadata placeholders
- `src/xai_demo_suite/` — reusable package code
- `tests/` — unit and integration tests

Use `docs/tasks/active/` for substantial work so another engineer or Codex
thread can resume from checked-in context.
