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

### MVTec LOCO AD
Use for:
- logical anomaly limitations
- structure versus logic comparisons

### Waterbirds
Use for:
- classic shortcut demonstration
- background reliance and counterfactual swaps

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
