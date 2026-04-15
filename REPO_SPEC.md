# XAI Demo Suite Repository Specification

## 1. Purpose

Build a clean, reusable, academically oriented repository for a **series of explainable AI demos** in vision, centred on:

- **model shortcuts and spurious correlations**
- **what a model has actually learned**
- **what evidence drove the decision**
- **what would change the decision**
- **when explanations fail or drift**
- **what PatchCore can and cannot do**

This repository is not a notebook dump and not a benchmark zoo. It is a **curated demo product** with a coherent technical story, a strong codebase, and a durable operating model for long Codex sessions.

The repository should help an audience answer questions such as:

- Is the model right for the right reason?
- Which evidence did it use?
- Which exemplars or prototypes shaped the result?
- What counterfactual change would flip the prediction or reduce the anomaly score?
- Does the explanation survive nuisance variation and shift?
- Where does PatchCore stop being the right tool?

## 2. Design philosophy

### 2.1 Demo philosophy

Every demo must tell a **complete engineering story**:

1. show a baseline success or apparent success;
2. reveal the hidden shortcut, pitfall, or limitation;
3. make the failure legible through explainability;
4. apply a concrete intervention;
5. re-test and show what improved, what did not, and what remains risky.

### 2.2 Repository philosophy

The codebase should be:

- **modular**, with nearly all logic in importable package code;
- **notebook-light**, with notebooks used for storytelling and visual analysis rather than business logic;
- **testable**, with unit, integration, and smoke tests;
- **typed**, with explicit Python type hints throughout;
- **well commented**, using **UK English** in comments, docs, and user-facing text;
- **Codex-friendly**, with short durable instructions and deeper checked-in specs;
- **anti-drift by design**, so long conversations do not become the only source of truth.

### 2.3 Explanation philosophy

The suite should standardise on four explanation questions across demos:

1. **Evidence**: what pixels, patches, regions, or features drove the output?
2. **Provenance**: which training examples, prototypes, or normal exemplars shaped the behaviour?
3. **Counterfactual change**: what small plausible change would alter the output?
4. **Stability**: does the explanation survive benign perturbation, retraining noise, and shift?

These four questions are the cross-demo contract.

## 3. Primary demonstration pillars

### 3.1 Pillar A: Shortcut Lab

Goal: demonstrate classification models that achieve good superficial performance while relying on a shortcut.

Primary use cases:

- **Waterbirds** or similar classic spurious-correlation classification
- an **industrial shortcut demo** using synthetic border, watermark, crop, lighting, or fixture artefacts
- optional natural-context shortcut examples such as MetaShift or Spawrious

Core questions:

- where did the classifier look?
- did it focus on the object or the nuisance?
- what happens when we remove or swap the nuisance?
- can we fix it with data, augmentation, explanation regularisation, or architecture?

Recommended model set:

- baseline classifier: ResNet or ViT
- explanation methods: Grad-CAM, Integrated Gradients, occlusion / perturbation
- optionally: ProtoPNet as an inherently more interpretable comparator

### 3.2 Pillar B: PatchCore Lab

Goal: make PatchCore the centrepiece for industrial anomaly detection explainability.

Core ideas to show:

- PatchCore learns **normality from nominal data only**
- anomaly score is based on distance to a memory bank of normal patch features
- explanation is not only an anomaly map; it is also **nearest normal exemplar evidence**
- real explanation should retrieve **actual source patches and source images**, not feature-space fantasies

Mandatory views for every selected example:

- input image
- anomaly map
- top anomalous patch
- top-k nearest normal patches
- full source images for those normal patches
- per-patch distance summary
- optional counterfactual patch replacement preview

### 3.3 Pillar C: PatchCore Limits Lab

Goal: show that PatchCore is strong, useful, and still bounded.

Three limitations must be explicit:

#### Count
PatchCore does not natively count distinct anomalous instances. It scores anomaly relative to normality, often via top anomalous patches rather than explicit object counting.

#### Severity
PatchCore does not natively produce a calibrated notion of engineering severity. A high feature-space distance is not automatically a high business or safety severity.

#### Semantic location / logic
PatchCore localises unusual regions, but does not natively produce symbolic statements such as:
- slot 3 is empty
- there are 5 objects instead of 6
- the left connector is rotated
- the component is present but in the wrong place

This lab should compare PatchCore with:
- **MVTec LOCO AD** examples
- optional **ComAD** or another logic-aware / component-aware method
- synthetic counting and logic boards that make these limitations visually obvious

### 3.4 Pillar D: Robustness and Explanation Drift Lab

Goal: show that prediction drift and explanation drift are different signals.

Use cases:

- lighting shift
- contrast / blur / compression corruption
- nuisance contamination in the normal set
- optional harder industrial benchmarks such as MVTec AD 2
- optional realism benchmark such as AutoVI

Questions:

- does the prediction stay the same while the explanation moves?
- does the anomaly map move to a nuisance region?
- do nearest exemplars become less semantically meaningful?
- can explanation drift expose a pipeline or acquisition change earlier than a headline metric?

## 4. Dataset plan

## 4.1 Core datasets

### Required
- **MVTec AD**
- **MVTec LOCO AD**
- **Waterbirds** or a similarly canonical spurious-correlation classification set

### Strong additions
- **MVTec AD 2**
- **MetaShift**
- **Spawrious**
- **VisA**
- **NEU** or **GC10-DET** for industrial shortcut classification

### Optional synthetic datasets generated inside the repo
- slot-board / assembly-logic generator
- counting board with missing, extra, or duplicated components
- severity generator with controlled scratch size, area, contrast, or deformation level
- nuisance injector for borders, corner stamps, lighting gradients, vignette, crop shifts, and camera-like artefacts

## 4.2 Data policy

The repository must **not** commit raw datasets. Instead it must provide:

- download scripts or adapters;
- checksums when feasible;
- per-dataset metadata:
  - licence
  - source URL
  - expected folder layout
  - academic / non-commercial restrictions
  - citation text
- a `data_registry.yaml` file describing each supported dataset;
- a `make data-*` or `uv run python -m ...` workflow for fetching and preparing data.

## 4.3 Recommended directory layout for data

```text
data/
  raw/           # downloaded exactly as supplied; never edited in place
  external/      # optional manually placed corpora
  interim/       # temporary conversions, caches, unpacked files
  processed/     # canonical processed format used by package code
  synthetic/     # generated datasets for shortcut and limitation demos
  artefacts/     # cached embeddings, memory banks, prototype stores, plots
```

Rules:

- never overwrite `data/raw/`;
- any processing step must read from raw/interim and write new outputs elsewhere;
- processed datasets must be reproducible from scripts;
- large generated artefacts should be safely re-creatable.

## 5. Demo catalogue

## 5.1 Demo 01 — Waterbirds shortcut

Narrative:
- train a baseline classifier;
- show apparently good performance;
- expose background reliance;
- run background swap / masked-background counterfactuals;
- fix via data rebalance or explanation penalty;
- compare explanations before and after.

Must include:
- worst-group metrics, not only overall accuracy;
- saliency view plus perturbation check;
- nearest training examples or failure slices.

## 5.2 Demo 02 — Industrial shortcut trap

Narrative:
- choose NEU or GC10-DET, or use a synthetic industrial classification set;
- inject a shortcut such as border, watermark, crop offset, or fixture mark;
- show that the model learns the nuisance;
- remove the nuisance or regularise it away;
- compare explanations before and after.

This demo should feel like the industrial analogue of wolves-and-snow.

## 5.3 Demo 03 — PatchCore on MVTec AD

Narrative:
- train PatchCore on a clean normal set;
- show successful anomaly localisation;
- show nearest normal patches as provenance;
- show what patch replacement or masking does to anomaly score.

This is the **hero demo**.

Preferred categories:
- aligned objects first;
- categories where source patch retrieval is visually interpretable.

## 5.4 Demo 04 — PatchCore learns the wrong normal

Narrative:
- contaminate nominal training images with a nuisance such as:
  - lighting gradient
  - corner tab
  - border
  - faint stamp
- rebuild PatchCore;
- demonstrate false positives or distorted explanations;
- show that the memory bank now encodes nuisance as part of normality;
- fix the pipeline and rebuild;
- compare the exemplar retrieval before and after.

This is the industrial analogue of a shortcut demo for anomaly detection.

## 5.5 Demo 05 — PatchCore cannot count

Narrative:
- create or use repeated-object layouts;
- vary the number of missing or extra components;
- show that the image-level anomaly score is not a principled count;
- optionally add a lightweight connected-component or instance layer on top as a pragmatic fix.

## 5.6 Demo 06 — PatchCore does not know severity

Narrative:
- generate controlled scratch / blob / defect severity sweeps;
- compare anomaly score against:
  - defect area
  - contrast
  - depth or width proxy
  - human-labelled severity bucket
- show that feature-space novelty and severity are not the same construct.

## 5.7 Demo 07 — PatchCore struggles with logical anomalies

Narrative:
- use MVTec LOCO AD;
- compare structural anomalies with logical anomalies;
- show that local patch novelty does not equal logical understanding;
- optionally compare with a logic-aware method.

## 5.8 Demo 08 — Explanation drift under shift

Narrative:
- choose one classifier and one anomaly detector;
- run lighting, blur, compression, and contrast shifts;
- track both performance drift and explanation drift;
- show examples where the metric is stable but the explanation has drifted.

## 6. Package architecture

```text
src/xai_demo_suite/
  __init__.py

  config/
    schemas.py
    registry.py
    loading.py

  data/
    base.py
    registry.py
    manifests.py
    downloaders/
    transforms/
    synthetic/

  models/
    classification/
    anomaly/
    patchcore/
    prototype/
    logic/

  explain/
    evidence.py
    provenance.py
    counterfactuals.py
    drift.py
    perturbation.py
    metrics.py

  demos/
    shortcut_lab/
    patchcore_lab/
    patchcore_limits/
    robustness_lab/

  reports/
    cards.py
    html.py
    notebook_exports.py
    artefacts.py

  vis/
    image_panels.py
    overlays.py
    dashboards.py
    plots.py

  eval/
    classification.py
    anomaly.py
    robustness.py
    notebook_smoke.py

  utils/
    io.py
    typing.py
    seeds.py
    logging.py
```

## 6.1 Architectural rules

- All reusable logic belongs in `src/`.
- Notebooks may call package functions but must not contain the only implementation of important logic.
- Demo-specific code should still use common abstractions where possible.
- Visualisation code should be centralised so plots look consistent across notebooks and reports.
- Config loading must be deterministic and typed.
- Dataset handling must be decoupled from model code.

## 7. Core abstractions

## 7.1 Demo contract

Each demo should implement a common contract, even if not all methods populate every field.

Suggested interfaces:

```python
class DemoRunner(Protocol):
    def run(self, config: DemoConfig) -> DemoResult: ...
```

```python
@dataclass(slots=True)
class DemoResult:
    name: str
    summary: str
    predictions: list[PredictionRecord]
    evidence: list[EvidenceArtefact]
    provenance: list[ProvenanceArtefact]
    counterfactuals: list[CounterfactualArtefact]
    stability: list[StabilityArtefact]
    metrics: dict[str, float]
    assets: dict[str, Path]
```

## 7.2 Explanation artefact contract

Create a standard schema so notebooks and dashboards can consume explanations consistently.

```python
@dataclass(slots=True)
class EvidenceArtefact:
    sample_id: str
    method: str
    target: str
    heatmap_path: Path | None
    mask_path: Path | None
    top_regions: list["RegionScore"]
```

```python
@dataclass(slots=True)
class ProvenanceArtefact:
    sample_id: str
    method: str
    reference_ids: list[str]
    reference_scores: list[float]
    reference_image_paths: list[Path]
    reference_boxes: list["BoundingBox"] | None
    note: str
```

```python
@dataclass(slots=True)
class CounterfactualArtefact:
    sample_id: str
    method: str
    description: str
    before_score: float
    after_score: float
    output_path: Path | None
```

```python
@dataclass(slots=True)
class StabilityArtefact:
    sample_id: str
    method: str
    perturbation_name: str
    prediction_shift: float
    explanation_shift: float
    note: str
```

## 8. PatchCore-specific implementation notes

## 8.1 What must be stored

During memory-bank creation, store metadata for every retained or full-bank patch:

- source image id
- source dataset split
- source file path
- patch coordinates in source image
- feature vector id
- optional RGB crop path
- optional precomputed thumbnail path

Without this metadata, exemplar-based explanation will be weak.

## 8.2 Two-stage retrieval for faithful explanations

Recommended approach:

1. use the usual coreset-reduced memory bank for ordinary PatchCore scoring;
2. for the most anomalous test patches, re-rank against the full stored nominal patch bank or full saved source metadata;
3. display top-k real nearest normal patches.

Reason:
the scoring bank and the explanation bank do not need to be identical if explanation fidelity improves and the distinction is made explicit.

## 8.3 PatchCore counterfactuals

Implement at least two simple counterfactual probes:

- **mask out** the top anomalous patch and recompute score;
- **replace** the top anomalous patch with the nearest normal patch and recompute score.

These are not “true” causal proofs, but they are excellent didactic probes.

## 8.4 PatchCore limitation tests

Create explicit test suites for:
- multi-instance anomalies with identical anomaly score order issues;
- severity ranking disagreement;
- logical anomalies;
- nuisance contamination of the nominal bank;
- shift in explanation region despite similar image-level score.

## 9. Notebook and narrative strategy

## 9.1 Notebook principles

Notebooks are for:
- exposition;
- visual storytelling;
- side-by-side comparisons;
- a small amount of orchestration.

Notebooks are not for:
- hidden model logic;
- long unstructured exploratory detours in the main branch;
- one-off helper code that should really live in `src/`.

## 9.2 Notebook layout

```text
notebooks/
  00_overview.ipynb
  01_waterbirds_shortcut.ipynb
  02_industrial_shortcut_trap.ipynb
  03_patchcore_mvtec_ad.ipynb
  04_patchcore_wrong_normal.ipynb
  05_patchcore_count_limit.ipynb
  06_patchcore_severity_limit.ipynb
  07_patchcore_loco_logic_limit.ipynb
  08_explanation_drift.ipynb
```

For version control hygiene:
- pair notebooks with Jupytext percent scripts or Markdown notebooks;
- keep outputs stripped in git;
- produce rendered artefacts as CI or release outputs.

## 9.3 Notebook template

Every notebook should follow the same top-level structure:

1. Title and learning goals
2. Why this demo matters
3. Dataset and task definition
4. Model and explanation methods
5. Baseline result
6. Failure or pitfall
7. Intervention
8. Re-test
9. What we learned
10. Residual risks and next questions

## 10. Documentation system of record

## 10.1 Main principle

Do not put the whole repository brain into one giant `AGENTS.md`.

Use:
- a **short AGENTS.md** as the map and working agreement;
- a structured `docs/` tree as the source of truth;
- `PLANS.md` and task specs as first-class long-running artefacts;
- ADRs for irreversible design decisions.

## 10.2 Required docs

```text
docs/
  ARCHITECTURE.md
  DATASETS.md
  STYLE_GUIDE.md
  TESTING.md
  DEMO_CATALOGUE.md
  PATCHCORE_NOTES.md
  XAI_CONTRACT.md
  CODEx_WORKFLOW.md
  runbooks/
    add_dataset.md
    add_demo.md
    cut_release.md
  tasks/
    TASK_TEMPLATE.md
  decisions/
    ADR-0001-demo-philosophy.md
    ADR-0002-notebook-policy.md
    ADR-0003-explanation-contract.md
```

## 10.3 ADR policy

Use ADRs for decisions such as:
- why Jupytext is used;
- why PatchCore is the hero anomaly model;
- why notebooks cannot contain unique business logic;
- why explanation artefacts have a standard schema;
- why certain datasets are in or out of scope.

## 11. Codex operating model

## 11.1 Why this matters

The repo must remain coherent through many iterations and long conversations. The anti-drift strategy is therefore as important as the code.

## 11.2 Durable Codex rules

Codex should be guided by:

- `~/.codex/AGENTS.md` for your personal defaults;
- repository `AGENTS.md` for shared repo behaviour;
- `docs/` as the system of record;
- `.agents/PLANS.md` for long tasks;
- `.agents/skills/` for reusable workflows;
- `.codex/agents/` for specialised subagents when the task benefits from separation of concerns.

## 11.3 Conversation control rules

### Rule 1: one thread per task
Use one Codex thread per bounded task, not one giant thread for the whole repository.

Examples of good task boundaries:
- “add Waterbirds data adapter”
- “implement PatchCore provenance browser”
- “write tests for anomaly map artefacts”
- “refactor demo 03 notebook to use shared figure builders”

### Rule 2: always start complex work from a checked-in plan
Any task larger than a modest one-file edit should begin with a plan document.

### Rule 3: task must name the source of truth
Every substantial Codex prompt should point to:
- the relevant spec page;
- the relevant ADRs;
- the task file;
- the done criteria.

### Rule 4: preserve hard-won workflows as skills
If you find yourself repeating a good prompt or correction, turn it into a skill.

### Rule 5: parallelise exploration, not conflicting edits
Use subagents or multiple threads for exploration and review, not for simultaneous edits to the same files unless isolated with worktrees.

## 11.4 Task packet format

Every serious task should have a checked-in task packet under `docs/tasks/active/` or similar.

Suggested sections:

- task id
- owner
- status
- background
- source docs
- scope
- out of scope
- technical constraints
- deliverables
- tests to add or update
- acceptance criteria
- open questions
- decision log

## 11.5 Definition of done for Codex tasks

A task is not done unless it includes, where relevant:

- code change
- tests
- docs updates
- notebook updates
- verification notes
- explicit note of remaining risk

## 12. Suggested custom agents

Only introduce custom agents if they clearly improve reliability or context control.

## 12.1 explorer
Purpose:
- read-heavy mapping of relevant files, symbols, docs, and execution paths
- no edits
- read-only sandbox

## 12.2 xai_architect
Purpose:
- translates spec into implementation plans
- checks architectural consistency across demos
- read-only by default

## 12.3 patchcore_worker
Purpose:
- implements PatchCore-specific modules, memory-bank metadata, and provenance features
- workspace write

## 12.4 evaluator
Purpose:
- runs targeted tests, notebook smoke checks, and small eval loops
- summarises regressions and missing coverage

## 12.5 docs_editor
Purpose:
- updates docs, ADRs, notebook markdown, and report text
- must preserve UK English style and repo terminology

## 12.6 reviewer
Purpose:
- code review against correctness, missing tests, documentation, and repo conventions
- read-only

## 13. Suggested skills

Keep skills narrow and trigger-specific.

## 13.1 `$demo-planner`
Use when:
- adding a new demo
- re-scoping an existing demo
- turning a fuzzy idea into a checked-in plan

Outputs:
- task packet
- affected files
- acceptance criteria
- risks

## 13.2 `$patchcore-explainer`
Use when:
- working on PatchCore visualisation, retrieval, or provenance
- checking whether explanation output includes real source evidence

Outputs:
- required artefacts
- retrieval checks
- UX guidance for plots and notebooks

## 13.3 `$notebook-polisher`
Use when:
- converting rough analysis into a narrative notebook
- aligning headings, markdown, figure ordering, and takeaways

Outputs:
- cleaner notebook flow
- markdown consistency
- explicit “what we learned” section

## 13.4 `$test-and-verify`
Use when:
- a task claims to be complete
- a change touches shared abstractions
- a notebook now depends on package code

Outputs:
- focused verification run
- list of executed commands
- remaining gaps

## 13.5 `$guidance-maintainer`
Use when:
- the same correction has been made twice
- AGENTS, skills, or docs need updating to encode a repeated lesson

Outputs:
- proposed updates to `AGENTS.md`, skills, or spec docs

## 14. Coding standards

## 14.1 Language and style

- Comments and docs must be written in **UK English**.
- Write complete docstrings for public functions, classes, and modules.
- Favour clarity over cleverness.
- Prefer small pure functions over large stateful helpers.
- Use explicit names rather than compressed shorthand.

## 14.2 Typing

- Use Python 3.12+ style type hints.
- Prefer `dataclass(slots=True)` or `TypedDict` / `Protocol` where appropriate.
- Public functions must have explicit parameter and return types.
- Run `mypy` in CI.

## 14.3 Testing

Minimum stack:
- `pytest`
- `pytest-cov`
- `mypy`
- `ruff`
- optional `nbmake` or notebook smoke runner
- optional property-based tests for small utilities

Test layers:
- unit tests for pure logic
- integration tests for data adapters and model wrappers
- smoke tests for at least one notebook per lab
- artefact tests for explanation objects and report generation

## 14.4 Reproducibility

- central seed control helper
- config objects serialised alongside outputs
- model checkpoints and artefacts saved with metadata
- figure-generation code deterministic where practical

## 15. Build and toolchain recommendation

Recommended Python tooling:

- `uv` for environment and dependency management
- `pyproject.toml` as the single build configuration file
- `ruff` for linting and formatting
- `mypy` for types
- `pytest` for tests
- `pre-commit` for local quality gates
- `jupytext` for notebook pairing
- `nbmake` or a notebook smoke runner in CI
- `pydantic` only where runtime validation genuinely helps, not by default everywhere

## 16. CI pipeline

Recommended CI stages:

1. install
2. lint
3. type-check
4. unit tests
5. integration smoke tests
6. notebook smoke tests on a reduced fixture dataset
7. docs link or structure checks

Optional nightly:
- larger dataset integration run
- sample notebook execution
- explanation drift regression smoke run

## 17. Release artefacts

Each major demo release should ship:

- notebook
- rendered HTML or static export
- selected output figures
- short demo card summarising:
  - task
  - model
  - explanation methods
  - key lesson
  - failure mode
  - intervention
  - remaining caveats

## 18. Minimum viable roadmap

## Phase 0 — Foundation
- scaffold repo
- set up toolchain
- create specs, AGENTS, plans, and task templates
- create one synthetic image fixture dataset for tests

## Phase 1 — Hero demo
- implement PatchCore lab on MVTec AD
- add real nearest-normal patch retrieval
- add notebook and report export
- add unit and smoke tests

## Phase 2 — Shortcut demos
- Waterbirds shortcut
- industrial shortcut trap
- saliency plus perturbation checks

## Phase 3 — Limits
- count demo
- severity demo
- LOCO logical anomaly demo

## Phase 4 — Robustness
- explanation drift lab
- nuisance contamination lab
- optional AD 2 integration

## 19. Risks to watch early

- notebooks becoming the real implementation;
- explanation screenshots without faithfulness checks;
- too many models, not enough coherent stories;
- too many large datasets before the core abstractions settle;
- long Codex threads replacing checked-in plans and docs;
- simultaneous threads editing the same files without worktree isolation;
- turning unstable workflows into automations too early.

## 20. Starter prompt patterns for Codex

### 20.1 New feature implementation
Read `AGENTS.md`, then read `docs/ARCHITECTURE.md`, `docs/XAI_CONTRACT.md`, and `docs/tasks/<TASK>.md`. Plan first. Do not edit anything until you can explain the file-level change set, the tests to update, and the done criteria. Then implement the smallest clean change that satisfies the task. Run the narrowest relevant lint, type, and test commands. Update docs if behaviour or structure changed.

### 20.2 New demo planning
Read `AGENTS.md`, `docs/DEMO_CATALOGUE.md`, `docs/PATCHCORE_NOTES.md` if relevant, and the task template. Create a plan for a new demo with narrative arc, datasets, models, explanation artefacts, tests, and notebook structure. Keep it aligned with the shared XAI contract and existing visual language.

### 20.3 Review request
Review this branch against `main`. Focus on correctness, explanation fidelity, missing tests, notebook drift, and whether the implementation still answers evidence, provenance, counterfactual, and stability questions.

## 21. Success criteria for the repository

The repository is successful if:

- a new engineer can understand the demo catalogue and repo layout quickly;
- Codex can implement bounded tasks without re-learning the project every session;
- notebooks are compelling but thin;
- the PatchCore demos are visually strong and technically honest;
- the shortcut demos make “right answer, wrong reason” obvious;
- the limitations demos make clear what anomaly detection does not solve;
- explanation artefacts are consistent enough to compare methods and datasets;
- the repo does not depend on one giant conversation to remain coherent.
