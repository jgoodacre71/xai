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
The flagship reports now also share a consistent static presentation chrome:
each one links back to the local hub, includes a presenter-facing demo brief,
and links onward to the next relevant demos in the suite.
It now surfaces the prepared local dataset state as well as the generated demo
order, so a fresh run is easier to inspect as a coherent suite.

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
path with configurable ResNet-18 tuning, worst-group metrics, Grad-CAM,
Integrated Gradients, context-masking perturbation checks, and a
prototype-exemplar comparator. When the prepared MetaShift manifest also
exists, the same report adds a natural-context extension section with the same
ERM-versus-group-balanced comparison. The synthetic proxy remains as the
fallback for fresh clones without local data.

NEU-CLS now provides the real industrial shortcut path for Demo 02 and Demo 08:

```bash
./.venv/bin/xai-demo-data fetch neu_cls --category shortcut_binary --dry-run
./.venv/bin/xai-demo-data fetch neu_cls --category shortcut_binary --archive-url <direct-archive-url>
./.venv/bin/xai-demo-data prepare neu_cls --category shortcut_binary
```

The fetch path is intentionally conservative. If the upstream page does not
give you a stable direct archive URL, place one archive under
`data/raw/neu_cls/archives/`, or point `prepare` at a manual source root with
`--source-root`.

KolektorSDD2 now provides a second real industrial shortcut path through the
same shared manifest contract:

```bash
./.venv/bin/xai-demo-data fetch ksdd2 --category shortcut_binary --dry-run
./.venv/bin/xai-demo-data fetch ksdd2 --category shortcut_binary --archive-url <direct-archive-url>
./.venv/bin/xai-demo-data prepare ksdd2 --category shortcut_binary
```

You can point either industrial report at the KSDD2 manifest explicitly:

```bash
./.venv/bin/xai-demo-report shortcut-industrial --real-manifest-path data/processed/ksdd2/shortcut_binary/manifest.jsonl
./.venv/bin/xai-demo-report explanation-drift --industrial-manifest-path data/processed/ksdd2/shortcut_binary/manifest.jsonl
```

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
available explicitly through dense ResNet-18 and WideResNet50-2 feature-map
extractors. It only uses pretrained weights when a pretrained feature-map
option is requested.

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

If the prepared MetaShift manifest exists at the default path, Demo 01 adds the
natural-context extension automatically. You can override that path explicitly
with `--metashift-manifest-path`.

You can force the fallback path with `--no-real-data`, or use random backbone
weights for quick local smoke tests with `--weights none`.

You can switch extractors explicitly:

```bash
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor mean_rgb
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor resnet18_random
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_random --coreset-size 512
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_pretrained --coreset-size 512
./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_wide_resnet50_2_pretrained --coreset-size 512
```

The pretrained commands may download Torchvision weights into the local Torch
cache. That cache, generated reports, and memory-bank artefacts are not tracked
by git.

The same report path can now be reused for other prepared MVTec AD categories.
For example, with capsule prepared locally:

```bash
./.venv/bin/xai-demo-report patchcore-bottle \
  --manifest-path data/processed/mvtec_ad/capsule/manifest.jsonl \
  --output-dir outputs/patchcore_capsule \
  --cache-path data/artefacts/patchcore/capsule/report_bank.npz \
  --feature-extractor feature_map_resnet18_pretrained \
  --max-train 20 \
  --max-examples 3 \
  --coreset-size 512 \
  --input-size 224 \
  --no-cache
```

Use `--benchmark-limit` for quick smoke runs; omit it to score the full local
MVTec AD bottle test split in the report diagnostics.

The narrative notebooks are checked in under `notebooks/` as output-free
`.ipynb` files paired with Jupytext-style percent scripts. The Demo 03 pair is
`notebooks/03_patchcore_mvtec_ad.ipynb` and `notebooks/03_patchcore_mvtec_ad.py`.
All notebook sources delegate the implementation to package code, and the test
suite now includes notebook smoke execution over reduced local configs.

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

Generate the shortcut demo with:

```bash
./.venv/bin/xai-demo-report shortcut-industrial
```

When the prepared NEU-CLS shortcut manifest exists, the report at
`outputs/shortcut_industrial/index.html` uses a curated real NEU
scratches-versus-inclusion shortcut slice with a correlated border stripe.
Otherwise it falls back to the synthetic shortcut generator. In both modes it
shows a learned shortcut model, a stamp-invariant intervention model, Grad-CAM
and Integrated Gradients overlays, and known-region perturbation diagnostics
over the shortcut region and the part.

When the prepared KSDD2 shortcut manifest exists, the same report can be
pointed at that second real industrial path with `--real-manifest-path`.

## Explanation Drift Report

Generate the synthetic drift demo with:

```bash
./.venv/bin/xai-demo-report explanation-drift
```

The report is written to `outputs/explanation_drift/index.html`. It separates
performance drift from explanation drift for the learned industrial shortcut
models under blur, contrast, compression, lighting, and shadow shifts. When the
prepared NEU-CLS manifest exists, the classifier section uses the same curated
real NEU scratches-versus-inclusion shortcut slice. When local MVTec bottle
data is prepared, it adds a PatchCore
anomaly-drift section with image-level AUC, top-patch movement, and
mask-coverage checks. When local MVTec AD 2 scenario manifests are prepared,
the same report now adds second-wave anomaly-drift sections for those
scenarios as well. When local VisA manifests are prepared, the same report adds
cross-dataset anomaly-drift sections there too.

## Review Pack

Generate the compact external-review pack with:

```bash
./.venv/bin/xai-demo-report review-pack
```

The review pack is written to `outputs/review_pack/index.html`. It gives a
single entry point for external reviewers or ChatGPT, with dataset-readiness
checks, best-entry links, demo-card summaries, caveats, and a handoff order
for repo docs and flagship screenshots.

## Main files

- `REPO_SPEC.md` — the long-form repository specification
- `AGENTS.md` — short always-on repo guidance for Codex
- `.agents/PLANS.md` — execution-plan template
- `.agents/skills/` — reusable workflow skills
- `.codex/agents/` — optional specialised subagents
- `docs/` — source-of-truth documentation skeleton
- `data_registry.yaml` — dataset metadata placeholders
- `src/xai_demo_suite/` — reusable package code
- `notebooks/` — output-free notebook showroom plus paired percent scripts
- `tests/` — unit and integration tests

Use `docs/tasks/active/` for substantial work so another engineer or Codex
thread can resume from checked-in context.
