# PatchCore notes

## What PatchCore is good at

- training from normal-only data
- strong anomaly localisation
- exemplar-style explainability when source patch metadata is preserved
- industrial anomaly triage on structural defects

## What PatchCore is not

- a calibrated severity model
- an object counter
- a symbolic reasoning system
- a logic checker for assembly constraints
- a causal explanation system

## Non-negotiable implementation rule

Store source image and patch coordinates for exemplar retrieval.

## Recommended visual contract

For each highlighted test image:
- show anomaly map;
- show top anomalous patch;
- show top-k nearest nominal patches;
- show full source images;
- note distance values;
- optionally show a patch replacement counterfactual.

## Recommended limitation demos

- count mismatch
- severity mismatch
- logical anomaly mismatch
- nuisance contamination of the normal bank
