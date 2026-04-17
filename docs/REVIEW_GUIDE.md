# Review Guide

This document is the compact but complete guide to what the repository now
contains, how it evolved, what each demo proves, and how to review it with
ChatGPT or a human reviewer.

## What this repo is

The repository is a local, presentation-first XAI demo suite built around four
flagship pillars:

1. shortcut learning in natural and industrial settings
2. PatchCore anomaly evidence with nearest-normal provenance
3. explicit PatchCore limits, including logic failures
4. prediction drift versus explanation drift under nuisance shifts

The design rule has stayed consistent throughout: reusable logic lives in
`src/xai_demo_suite/`, notebooks stay thin, generated outputs stay local, and
the repo memory sits in checked-in docs under `docs/`.

## What was built throughout

The work progressed in four broad phases.

### 1. Foundation and repo structure

- Created the package skeleton, CLI layer, report builders, tests, and dataset
  registry.
- Established the rule that notebooks are narrative wrappers over `src/`, not
  the implementation layer.
- Added task memory under `docs/tasks/completed/` so the repo has a durable
  change trail.

### 2. First flagship: serious local PatchCore

- Added real MVTec AD support and a visible Demo 03 report.
- Built dense feature-map PatchCore extractors with pretrained and random
  backbones.
- Added deterministic coreset reduction and cache management.
- Added nearest-normal provenance retrieval and report-level benchmark
  diagnostics.
- Extended the same report path to a stronger pretrained WideResNet50-2 option
  and a second local category path through capsule.

### 3. Broader demo suite and real-data upgrades

- Added real Waterbirds support for Demo 01, with ERM versus group-balanced
  comparisons, Grad-CAM, Integrated Gradients, perturbation checks, and a
  prototype-style comparator.
- Added MetaShift as a natural-context extension for the shortcut-learning
  story.
- Reworked Demo 02 and Demo 08 around a learned industrial classifier path.
- Added real industrial support through NEU-CLS and then a second industrial
  adapter through KolektorSDD2.
- Added MVTec LOCO AD for Demo 07 and a narrow component-aware comparator to
  make the logic limitation explicit.
- Added second-wave dataset support for MVTec AD 2 and VisA.

### 4. Presentation and reviewability

- Added the local demo hub at `outputs/index.html`.
- Added presenter-facing chrome to the four flagship reports.
- Added the review pack at `outputs/review_pack/index.html`.
- Added the full notebook suite under `notebooks/` with paired percent-script
  sources and notebook smoke checks.

## The best way to review the repo

If you are reviewing locally, use this order:

1. open `outputs/review_pack/index.html`
2. open `outputs/index.html`
3. review the four flagship reports in this order:
   - Demo 03
   - Demo 01
   - Demo 07
   - Demo 08
4. then inspect Demo 02, Demo 04, Demo 05, and Demo 06
5. finish with:
   - `README.md`
   - `docs/DEMO_STATUS.md`
   - `docs/DEMO_CATALOGUE.md`
   - `docs/DATASETS.md`
   - `docs/PATCHCORE_NOTES.md`

That order matches the strongest story in the suite: first show what works,
then show what fails, then show how explanations move under shift.

## How to let ChatGPT examine this repo

There are two sensible paths.

### Best path: GitHub connector

Use this when the repo is on GitHub and your ChatGPT plan/exposure supports the
GitHub app or connector.

1. push the repo to GitHub
2. in ChatGPT, connect GitHub in Settings
3. authorize the repository
4. ask ChatGPT to inspect the repo, starting with:
   - `README.md`
   - `REPO_SPEC.md`
   - `AGENTS.md`
   - `docs/REVIEW_GUIDE.md`
   - `docs/DEMO_STATUS.md`
   - `docs/DEMO_CATALOGUE.md`
   - `docs/DATASETS.md`
5. then point it at the key generated outputs and screenshots

Why this is the best route:
- ChatGPT can reason over the live repo structure and cite the code and docs
- it avoids losing context across many manual uploads
- it is easier to ask follow-up code questions

### Fallback path: ChatGPT Project plus file uploads

Use this when the GitHub connector is not available in your ChatGPT
experience.

1. create a Project in ChatGPT for this repo
2. upload the repo docs first:
   - `README.md`
   - `REPO_SPEC.md`
   - `AGENTS.md`
   - `docs/REVIEW_GUIDE.md`
   - `docs/DEMO_STATUS.md`
   - `docs/DEMO_CATALOGUE.md`
   - `docs/DATASETS.md`
3. then upload screenshots or PDF exports of the flagship reports
4. ask ChatGPT to assess:
   - spec coverage
   - demo quality
   - model/data strength
   - presentation clarity
   - remaining gaps

Why this is second-best:
- it works without connector support
- but plain chat handles codebases less cleanly than a live GitHub connection
- HTML outputs are less convenient than docs, PDFs, and screenshots

## Suggested ChatGPT prompts

Use prompts shaped like these:

- "Review this repository against `REPO_SPEC.md`. Tell me which parts are
  strong, which are weak, and which claims are not yet fully justified."
- "Read `docs/REVIEW_GUIDE.md`, `docs/DEMO_STATUS.md`, and the uploaded report
  screenshots. Tell me whether the suite feels like a coherent XAI product or a
  bag of demos."
- "Focus on Demo 03 and Demo 07. Do the current data, model choices, and
  report narratives make the intended claims honestly?"
- "Review the generated outputs as a presentation product. What would you cut,
  reorder, or tighten for a live academic demo?"

## Demo-by-demo guide

### Demo 01: Waterbirds shortcut

**What it shows**

The model can achieve strong average accuracy while still relying on the wrong
evidence. Background can carry a spurious shortcut, and group-balanced training
improves the worst-group story even when overall accuracy looks acceptable.

**Data**

- real path: Waterbirds
- extension: MetaShift cat-versus-dog indoor/outdoor context split
- fallback: synthetic proxy

**Model and XAI**

- frozen or partially tuned ResNet-18 classifier path
- ERM versus group-balanced linear probe comparison
- Grad-CAM
- Integrated Gradients
- context masking perturbation checks
- prototype-style exemplar comparator

**Main claim**

Average accuracy can hide shortcut dependence. Attribution and subgroup metrics
make that visible.

**What to say live**

"The baseline looks acceptable if you only read headline accuracy. The group
breakdown and explanation overlays show it is still looking in the wrong place.
The intervention improves the failure mode, not just the score."

**Main caveat**

This is a strong didactic shortcut demo, but it is still based on a compact
local fine-tuning path rather than a heavy benchmark training regime.

### Demo 02: Industrial shortcut trap

**What it shows**

A classifier can latch onto fixture leakage, stamps, or border cues instead of
surface condition. The intervention helps, but does not magically solve the
problem.

**Data**

- real path: curated NEU scratches-versus-inclusion shortcut slice
- second real option: KolektorSDD2 shared shortcut manifest
- fallback: synthetic industrial proxy

**Model and XAI**

- learned convolutional industrial classifier
- clean versus challenge splits
- shortcut-randomised intervention model
- known-region diagnostics

**Main claim**

Real industrial shortcuts can be made visible, measured, and partially reduced
with targeted intervention, but the problem is not trivial.

**What to say live**

"The clean score is misleading. Once I swap or remove the correlated cue, the
baseline collapses. The intervention recovers some of that, which is the real
point of the demo."

**Main caveat**

This is stronger than the original toy version, but still not a broad
industrial benchmark family.

### Demo 03: PatchCore on MVTec AD

**What it shows**

PatchCore can produce not only an anomaly map but also a provenance story:
which nearest normal patches make the anomaly look unusual. This is the hero
demo in the repo.

**Data**

- real path: MVTec AD bottle
- extended local category path: MVTec AD capsule

**Model and XAI**

- deterministic local feature baseline
- dense ResNet-18 feature-map path
- dense WideResNet50-2 feature-map path
- coreset reduction
- nearest-normal provenance retrieval
- benchmark diagnostics on the local test split

**Main claim**

PatchCore is most compelling when it explains anomaly evidence as local novelty
relative to specific nominal exemplars.

**What to say live**

"This is not just a heatmap. I can show you the anomalous patch and the most
similar normal patch it failed to match. That makes the anomaly score legible."

**Main caveat**

This is a serious local implementation, not an official benchmark
reproduction.

### Demo 04: PatchCore learns the wrong normal

**What it shows**

If the nominal memory bank is contaminated with nuisance patterns, PatchCore
can learn the wrong notion of normality and produce false positives or false
comfort.

**Data**

- synthetic slot-board construction

**Model and XAI**

- PatchCore-style memory-bank comparison
- clean versus contaminated nominal set comparison
- provenance comparison

**Main claim**

PatchCore explanations are only as trustworthy as the notion of normality they
are anchored to.

**What to say live**

"The model is being faithful here. The failure is not that the heatmap lies. It
is that the memory bank has learned the wrong reference world."

### Demo 05: PatchCore limits lab

**What it shows**

Patch novelty is not the same thing as count, relational structure, or logic.

**Data**

- synthetic slot boards

**Model and XAI**

- PatchCore-style novelty scoring
- repeated-object layouts

**Main claim**

You cannot recover symbolic count from local novelty alone.

**What to say live**

"PatchCore is telling you where something looks unfamiliar, not whether the
scene obeys a counting rule."

### Demo 06: PatchCore severity mismatch

**What it shows**

Novelty score and semantic severity are not the same quantity.

**Data**

- synthetic scratch-severity sweep

**Model and XAI**

- PatchCore-style scoring against controlled defect area

**Main claim**

A more severe defect is not guaranteed to receive a more meaningful anomaly
score in the way an operator expects.

**What to say live**

"The score answers novelty, not business severity. Those are different
questions."

### Demo 07: PatchCore logical anomaly limits

**What it shows**

PatchCore can localise novelty on MVTec LOCO AD, but logic-aware or
component-aware checks are still needed for packaging or rule violations.

**Data**

- real path: MVTec LOCO AD `juice_bottle`
- fallback: synthetic proxy

**Model and XAI**

- PatchCore-style local anomaly evidence
- category-specific front-label rule comparator

**Main claim**

PatchCore is useful for local novelty, but logic requires a different modelling
layer.

**What to say live**

"PatchCore can tell you something local changed. It cannot on its own express a
rule like 'the front label must be present and aligned'."

**Main caveat**

The current comparator is intentionally narrow and category-specific.

### Demo 08: Explanation drift under shift

**What it shows**

Prediction drift and explanation drift are related but not identical. A model
can keep predicting the same class while its evidence moves, or lose accuracy
while keeping a superficially similar attribution pattern.

**Data**

- classifier path: synthetic industrial proxy or real NEU / KSDD2 shortcut path
- anomaly path: MVTec AD bottle
- extended anomaly sections: MVTec AD 2 and VisA when prepared

**Model and XAI**

- learned industrial shortcut classifier
- blur, contrast, compression, lighting, and shadow perturbations
- optional PatchCore anomaly-drift sections

**Main claim**

Explanation stability should be checked explicitly; it is not guaranteed by
stable predictions.

**What to say live**

"Even when the top-line prediction looks stable, the model may be using
different evidence under shift. That matters for trust."

## Current strongest reports

These are the outputs to show first:

1. `outputs/patchcore_bottle/index.html`
2. `outputs/patchcore_bottle_wrn50/index.html`
3. `outputs/patchcore_capsule/index.html`
4. `outputs/waterbirds_shortcut/index.html`
5. `outputs/patchcore_logic/index.html`
6. `outputs/explanation_drift/index.html`

## Current limitations to state honestly

- Demo 03 is strong, but it is still not a full benchmark-faithful PatchCore
  reproduction.
- Demo 02 and Demo 08 now use real industrial data paths, but the industrial
  story is still narrower than the natural-image shortcut story.
- Demo 07 makes the logical limitation clear, but the current comparator is
  deliberately narrow rather than a general logic model.
- Some reports remain more didactic than benchmark-driven by design.

## Reviewer checklist

Use this when assessing whether the repo is ready to show:

- Is the suite coherent as one product, not just eight independent reports?
- Do the flagship demos each make a clean, defensible point?
- Are the limitations framed honestly rather than oversold?
- Do the real-data paths dominate the most important demos?
- Is the generated presentation layer good enough for a live walkthrough?
- Are the docs sufficient for a third party or another model to review the repo
  without re-discovering its structure from scratch?
