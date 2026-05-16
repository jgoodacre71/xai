# Data requirements

This project separates generated control demos from real-data demos.
Generated demos must run without external files. Real-data demos must fail
clearly when their required manifests are missing.

## Demo data requirements

| Demo | Role | Data mode | External data required | Expected local manifest |
| --- | --- | --- | --- | --- |
| 00 Moons/Stars Clever-Hans two-act laboratory | No-permission behavioural XAI opener: Act I position shortcut, Act II near-invisible background/acquisition cue | `generated_controlled_demo` | No | Not applicable |
| 02 Industrial side-band marker shortcut | Real industrial shortcut with controlled side-band leakage | `real_neu_controlled_shortcut` | Yes | `data/processed/neu_cls/shortcut_binary/manifest.jsonl` |
| 01 Waterbirds shortcut audit | Literature-aligned natural shortcut benchmark | `real` | Yes | `data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl` |
| 03 PatchCore anomaly provenance | Anomaly maps plus nearest-normal provenance | Real data when prepared | Yes for real MVTec path | `data/processed/mvtec_ad/<category>/manifest.jsonl` |
| 04 PatchCore learns the wrong normal | Normal-set contamination and provenance failure | Generated / local controlled | No for current controlled path | Not applicable for current generated path |
| 05 PatchCore count limits | Repeated-object counting limitation | Generated / local controlled | No for current controlled path | Not applicable |
| 06 PatchCore severity limits | Novelty score versus severity mismatch | Generated / local controlled | No for current controlled path | Not applicable |
| 07 PatchCore logic limits | Logical anomaly limits and optional LOCO comparison | Real data when prepared | Yes for LOCO path | `data/processed/mvtec_loco_ad/<category>/manifest.jsonl` |
| 08 Explanation drift | Prediction and explanation drift under nuisance shift | Real data when prepared | Yes for real industrial/anomaly paths | NEU, KSDD2, MVTec, MVTec AD 2, or VisA manifests |
| 90 IEEE dataset scouting | Dataset candidate register | Registry only | No selected dataset yet | Optional `data/ieee_candidates.yaml` |

## Notebook data-status standard

Every active notebook should expose these fields near the top:

- `DEMO`
- `DATA_MODE`
- `EXTERNAL_DATA_REQUIRED`
- `MANIFEST_PATH`
- `MANIFEST_EXISTS`
- `PROJECT_ROOT`
- `DATASET_SOURCE`
- `LICENCE_NOTE`
- `MISSING_FILES`
- `SEED`

Generated demos should state:

- `DATA_MODE: generated_controlled_demo`
- `EXTERNAL_DATA_REQUIRED: false`

Real-data demos should fail clearly when required data are missing. They should
not silently fall back to generated data unless the notebook is explicitly a
generated-data demo.

## PatchCore data gates

PatchCore demos should support MVTec AD, MVTec LOCO AD, and MVTec AD 2 when
those datasets are permitted locally. If MVTec is unavailable or not permitted
at work, use VisA or approved IEEE anomaly data.

Do not treat MVTec as the only future option. The model-behaviour story is
anomaly evidence, nearest-normal provenance, and residual risk; the specific
dataset should be selected through the permission matrix.
