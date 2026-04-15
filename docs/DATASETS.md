# Datasets

## Policy

- Raw data is never committed.
- Dataset scripts must preserve source structure where feasible.
- Licences and usage restrictions must be recorded.
- Every dataset adapter must define a canonical processed representation.

## Required first-wave datasets

### MVTec AD
Use for:
- PatchCore baseline
- exemplar retrieval
- nuisance contamination experiments

Source:
- official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- official downloads page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads>

Licence:
- Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0)
- non-commercial use only

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_ad --category bottle --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_ad --category bottle
./.venv/bin/python -m xai_demo_suite.cli.data prepare mvtec_ad --category bottle
```

Storage policy:
- archives are downloaded to `data/raw/mvtec_ad/archives/`;
- extracted copies are written to `data/interim/mvtec_ad/`;
- manifests are written to `data/processed/mvtec_ad/<category>/manifest.jsonl`;
- raw archives, extracted data, and processed manifests are local artefacts and
  are excluded from git.

The full dataset is about 4.9 GB. Prefer one category first while developing the
PatchCore pipeline.

### MVTec LOCO AD
Use for:
- logical anomaly limitations
- structure versus logic comparisons

### Waterbirds
Use for:
- classic shortcut demonstration
- background reliance and counterfactual swaps

Current local status:
- Demo 01 has a generated Waterbirds-style proxy under
  `outputs/waterbirds_shortcut/synthetic/`.
- The real Waterbirds dataset is not sourced yet and must not be committed when
  added.

## Strong second-wave datasets

- MVTec AD 2
- MetaShift
- Spawrious
- VisA
- NEU / GC10-DET

## Synthetic generators to build in-repo

### Nuisance injector
Inject:
- border
- stamp
- habitat background
- corner tab
- lighting gradient
- crop shift
- vignette

### Count generator
Generate repeated objects with controlled missing/extra instances.

### Severity generator
Generate controlled defect intensity levels for showing the severity mismatch problem.

### Logic board generator
Generate slot-based arrangements with valid and invalid configurations.
