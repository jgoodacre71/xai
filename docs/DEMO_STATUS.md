# Demo status

This file summarises the current generated demo suite. It is intentionally
short; detailed task history lives under `docs/tasks/completed/`.

## Runnable commands

Generate the synthetic suite:

```bash
./.venv/bin/xai-demo-report suite
```

Generate the full local suite when MVTec AD bottle has been prepared:

```bash
./.venv/bin/xai-demo-report suite --include-mvtec
```

Generate the stronger local presentation suite with pretrained feature-map
PatchCore for Demo 03:

```bash
./.venv/bin/xai-demo-report suite \
  --include-mvtec \
  --mvtec-feature-extractor feature_map_resnet18_pretrained \
  --mvtec-max-train 20 \
  --mvtec-max-examples 3 \
  --mvtec-coreset-size 512 \
  --mvtec-input-size 224
```

Verify generated reports, cards, selected figures, and the local index:

```bash
./.venv/bin/xai-demo-report verify
```

## Current demos

| Demo | Status | Default output |
| --- | --- | --- |
| Demo 01 - Waterbirds Shortcut | Working real-data report when Waterbirds is prepared, with synthetic fallback and optional MetaShift extension | `outputs/waterbirds_shortcut/index.html` |
| Demo 02 - Industrial Shortcut Trap | Working real-data report when NEU-CLS is prepared, with synthetic fallback | `outputs/shortcut_industrial/index.html` |
| Demo 03 - PatchCore on MVTec AD bottle | Working local MVTec report with deterministic and explicit pretrained feature-map paths | `outputs/patchcore_bottle/index.html` |
| Demo 04 - PatchCore Learns the Wrong Normal | Working synthetic report | `outputs/patchcore_wrong_normal/index.html` |
| Demo 05 - PatchCore Limits Lab | Working synthetic report | `outputs/patchcore_limits/index.html` |
| Demo 06 - PatchCore Severity Mismatch | Working synthetic report | `outputs/patchcore_severity/index.html` |
| Demo 07 - PatchCore Logical Anomaly Limits | Working local MVTec LOCO report when data is prepared, with synthetic fallback | `outputs/patchcore_logic/index.html` |
| Demo 08 - Explanation Drift Under Shift | Working learned drift report with optional local MVTec anomaly section | `outputs/explanation_drift/index.html` |

The local `outputs/index.html` is a static presentation index with one tile per
demo, selected figures, report links, demo-card links, key lessons,
interventions, caveats, and a prepared-dataset summary for the current machine.
The four flagship reports now also share a presenter-facing header, demo-brief
section, and related-demo links so the suite reads like one coherent local demo
product rather than disconnected HTML pages.
The notebook layer now mirrors the same structure: each flagship notebook is
checked in as an output-free `.ipynb` plus a paired percent script under
`notebooks/`, and reduced notebook smoke runs are part of the local test slice.

## Known gaps

- Demo 01 now has a real Waterbirds path with configurable ResNet-18 tuning,
  worst-group metrics, Grad-CAM, Integrated Gradients, perturbation checks, and
  a prototype-exemplar comparator. When the prepared MetaShift manifest exists,
  the same report adds a natural-context extension on the cat-vs-dog
  indoor/outdoor split. The synthetic proxy remains as the fallback when the
  manifest is absent.
- Demo 02 now uses a prepared NEU-CLS-derived binary shortcut manifest when it
  exists, with synthetic fallback for fresh clones. The report includes
  intervention training, Grad-CAM, Integrated Gradients, and known-region
  shortcut diagnostics on either path.
- Demo 08 now uses learned industrial classifier drift under blur, contrast,
  compression, lighting, and shadow shifts, and it now switches to the prepared
  NEU-CLS manifest for the classifier path when available. It also adds an
  optional local PatchCore anomaly-drift section when MVTec bottle data is
  prepared. When MVTec AD 2 scenario manifests are prepared, the same report
  now adds second-wave anomaly sections for those scenarios. When VisA manifests
  are prepared, it adds cross-dataset anomaly-drift sections there too.
- Demo 07 now adds a category-specific front-label template comparator on local
  MVTec LOCO AD `juice_bottle`, so the report can contrast PatchCore patch
  novelty with a narrow packaging-rule check. Broader category coverage remains
  future work.
- MVTec AD 2 now has local dataset support and optional Demo 08 anomaly-drift
  sections when prepared.
- VisA now has local dataset support and optional Demo 08 anomaly-drift
  sections when prepared.
- Demo 03 now has an explicit pretrained ResNet-18 feature-map path, but it is
  still not an official PatchCore benchmark reproduction. Its benchmark panel
  reports local image-level max-patch AUC and top-patch mask diagnostics.
- The repository has no configured Git remote in this checkout.
