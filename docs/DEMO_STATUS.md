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
| Demo 01 - Waterbirds Shortcut Proxy | Working synthetic report | `outputs/waterbirds_shortcut/index.html` |
| Demo 02 - Industrial Shortcut Trap | Working synthetic report | `outputs/shortcut_industrial/index.html` |
| Demo 03 - PatchCore on MVTec AD bottle | Working local MVTec report with deterministic and explicit pretrained feature-map paths | `outputs/patchcore_bottle/index.html` |
| Demo 04 - PatchCore Learns the Wrong Normal | Working synthetic report | `outputs/patchcore_wrong_normal/index.html` |
| Demo 05 - PatchCore Limits Lab | Working synthetic report | `outputs/patchcore_limits/index.html` |
| Demo 06 - PatchCore Severity Mismatch | Working synthetic report | `outputs/patchcore_severity/index.html` |
| Demo 07 - PatchCore Logical Anomaly Limits | Working local MVTec LOCO report when data is prepared, with synthetic fallback | `outputs/patchcore_logic/index.html` |
| Demo 08 - Explanation Drift Under Shift | Working synthetic report | `outputs/explanation_drift/index.html` |

The local `outputs/index.html` is a static presentation index with one tile per
demo, selected figures, report links, demo-card links, key lessons,
interventions, and caveats.

## Known gaps

- Demo 01 still needs a real Waterbirds or equivalent shortcut dataset path;
  the current report is a synthetic proxy.
- Demo 07 currently uses one local MVTec LOCO AD category when prepared; broader
  category coverage and a component-aware comparator remain future work.
- Demo 03 now has an explicit pretrained ResNet-18 feature-map path, but it is
  still not an official PatchCore benchmark reproduction.
- The repository has no configured Git remote in this checkout.
