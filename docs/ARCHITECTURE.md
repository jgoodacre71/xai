# Architecture

This repository is organised around a simple principle:

**module code is the product; notebooks are the showroom.**

## Layers

### 1. Data layer
Handles:
- dataset download / registration;
- dataset metadata and licences;
- processing pipelines;
- synthetic dataset generation.

### 2. Model layer
Holds:
- classifiers for shortcut demos;
- PatchCore and related anomaly models;
- optional prototype or logic-aware models.

### 3. Explainability layer
Holds reusable interfaces for:
- evidence maps;
- provenance retrieval;
- counterfactual probes;
- stability and drift analysis.

### 4. Reporting and visual layer
Builds:
- image panels;
- HTML reports;
- demo cards;
- notebook-friendly figures.

### 5. Demo layer
Thin orchestration and per-demo configuration.

## Mandatory rule

If the same logic is needed by:
- more than one notebook,
- a notebook and a test,
- a notebook and a report,
- or more than one demo,

it belongs in `src/`.
