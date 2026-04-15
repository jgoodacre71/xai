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

## Current implementation status

The first implementation slice is a provenance-focused PatchCore baseline in
`src/xai_demo_suite/models/patchcore/`.

It deliberately uses mean RGB patch features rather than deep backbone features.
This is not the final PatchCore model. Its purpose is to make the memory-bank
contract testable:
- every retained nominal patch has a source image id;
- every retained nominal patch has source coordinates;
- scoring returns nearest normal patch evidence;
- nearest-normal evidence can be converted into the shared
  `ProvenanceArtefact` contract.

Deep features, coreset selection, anomaly-map rendering, and counterfactual
patch replacement should build on this provenance shape rather than bypass it.

## Feature extraction boundary

PatchCore-style scoring now accepts a `PatchFeatureExtractor` protocol. The
mean RGB extractor is the default deterministic baseline, while the optional
Torch/Torchvision extractor is a dependency boundary for future deep-feature
work.

Current rules:
- feature extractors must return one feature row per requested patch box;
- memory banks store the extractor `feature_name`;
- scoring rejects extractor/memory-bank mismatches;
- patch metadata is independent of extractor choice.

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
