# XAI contract

Every serious demo in this repository should aim to answer the same four questions.

## 1. Evidence
What image regions, patches, or features materially drove the output?

## 2. Provenance
Which examples, prototypes, or nominal exemplars shaped the behaviour?

## 3. Counterfactual change
What small plausible change would change the output or reduce the anomaly score?

## 4. Stability
Does the explanation remain meaningfully similar under benign perturbation, retraining noise, or domain shift?

## Why this matters

Without this shared contract, the repository becomes a grab-bag of unrelated explainability pictures.

With it, different demos can be compared honestly:
- classifier versus anomaly detector;
- saliency versus prototype versus exemplar retrieval;
- structural versus logical anomaly cases;
- clean versus shifted conditions.
