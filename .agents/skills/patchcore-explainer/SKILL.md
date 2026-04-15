---
name: patchcore-explainer
description: Work on PatchCore explainability and provenance. Use when implementing anomaly-map visualisation, nearest-normal patch retrieval, source-image patch metadata, counterfactual probes, or notebooks that explain what PatchCore compared against.
---

# PatchCore Explainer

## Purpose

Keep PatchCore explanation work faithful, visual, and operationally useful.

## Rules

- Preserve source image ids and patch coordinates.
- Prefer real nearest nominal patches over imagined reconstructions.
- Separate scoring logic from display logic, but keep their relationship explicit.
- If the memory bank is coreset-reduced, be clear whether displayed exemplars come from the coreset or from a full-bank re-rank.

## Required checks

For any substantial PatchCore explanation change, verify:
- source metadata is stored and recoverable;
- top-k retrieval is deterministic enough for tests;
- notebooks can show full source images and patch boxes;
- docs reflect what is and is not a faithful explanation.
