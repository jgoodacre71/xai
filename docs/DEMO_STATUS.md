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

Verify generated reports, cards, selected figures, and the local index:

```bash
./.venv/bin/xai-demo-report verify
```

## Current demos

| Demo | Status | Default output |
| --- | --- | --- |
| Demo 01 - Waterbirds Shortcut Proxy | Working synthetic report | `outputs/waterbirds_shortcut/index.html` |
| Demo 02 - Industrial Shortcut Trap | Working synthetic report | `outputs/shortcut_industrial/index.html` |
| Demo 03 - PatchCore on MVTec AD bottle | Working local MVTec report when data is prepared | `outputs/patchcore_bottle/index.html` |
| Demo 04 - PatchCore Learns the Wrong Normal | Working synthetic report | `outputs/patchcore_wrong_normal/index.html` |
| Demo 05 - PatchCore Limits Lab | Working synthetic report | `outputs/patchcore_limits/index.html` |
| Demo 06 - PatchCore Severity Mismatch | Working synthetic report | `outputs/patchcore_severity/index.html` |
| Demo 07 - PatchCore Logical Anomaly Limits | Working synthetic proxy report | `outputs/patchcore_logic/index.html` |
| Demo 08 - Explanation Drift Under Shift | Working synthetic report | `outputs/explanation_drift/index.html` |

## Known gaps

- Demo 01 still needs a real Waterbirds or equivalent shortcut dataset path;
  the current report is a synthetic proxy.
- Demo 07 still needs a real MVTec LOCO AD comparison; the current report is a
  synthetic logical-anomaly proxy.
- MVTec LOCO AD has a fetch/prepare workflow, but Demo 07 does not yet use real
  LOCO examples.
- The PatchCore hero report still uses deterministic local patch features rather
  than pretrained multi-scale PatchCore.
- The repository has no configured Git remote in this checkout.
