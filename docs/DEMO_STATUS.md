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
| Demo 02 - Industrial Shortcut Trap | Working synthetic report | `outputs/shortcut_industrial/index.html` |
| Demo 03 - PatchCore on MVTec AD bottle | Working local MVTec report when data is prepared | `outputs/patchcore_bottle/index.html` |
| Demo 04 - PatchCore Learns the Wrong Normal | Working synthetic report | `outputs/patchcore_wrong_normal/index.html` |
| Demo 05 - PatchCore Limits Lab | Working synthetic report | `outputs/patchcore_limits/index.html` |
| Demo 08 - Explanation Drift Under Shift | Working synthetic report | `outputs/explanation_drift/index.html` |

## Known gaps

- Demo 01 still needs a real Waterbirds or equivalent shortcut dataset path.
- Demo 06 and Demo 07 are represented inside the current limits report, but do
  not yet have dedicated reports.
- MVTec LOCO AD is not sourced yet.
- The PatchCore hero report still uses deterministic local patch features rather
  than pretrained multi-scale PatchCore.
- The repository has no configured Git remote in this checkout.
