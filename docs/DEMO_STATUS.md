# Demo status

This file summarises the current active demo surface. It is intentionally
short; detailed task history lives under `docs/tasks/completed/`.

## Runnable notebooks

Open and run the notebooks directly:

- `notebooks/overview/00_overview.ipynb`
- `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb`
- `notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb`
- `notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb`
- `notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb`
- `notebooks/patchcore_limits/05_patchcore_count_limit.ipynb`
- `notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb`
- `notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb`
- `notebooks/robustness_drift/08_explanation_drift.ipynb`

## Current demos

| Demo | Status | Active notebook |
| --- | --- | --- |
| Demo 01 - Waterbirds Shortcut | Working self-contained notebook demo | `notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb` |
| Demo 02 - Industrial Shortcut Trap | Working self-contained notebook demo | `notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb` |
| Demo 03 - PatchCore on MVTec AD bottle | Working self-contained notebook demo | `notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb` |
| Demo 04 - PatchCore Learns the Wrong Normal | Working self-contained notebook demo | `notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb` |
| Demo 05 - PatchCore Count Limit | Working self-contained notebook demo | `notebooks/patchcore_limits/05_patchcore_count_limit.ipynb` |
| Demo 06 - PatchCore Severity Mismatch | Working self-contained notebook demo | `notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb` |
| Demo 07 - PatchCore Logical Anomaly Limits | Working self-contained notebook demo | `notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb` |
| Demo 08 - Explanation Drift Under Shift | Working self-contained notebook demo | `notebooks/robustness_drift/08_explanation_drift.ipynb` |

The notebooks are grouped by storyline directories under `notebooks/`, checked
in output-free, and covered by reduced notebook smoke runs in the local test
slice.
The repository now also has a GitHub Actions CI workflow plus explicit
integration tests for the public CLI entry points.

## Known gaps

- Demo 01 now has a real Waterbirds path with configurable ResNet-18 tuning,
  worst-group metrics, Grad-CAM, Integrated Gradients, perturbation checks, and
  a prototype-exemplar comparator. When the prepared MetaShift manifest exists,
  the same report adds a natural-context extension on the cat-vs-dog
  indoor/outdoor split. The synthetic proxy remains as the fallback when the
  manifest is absent.
- Demo 02 now uses a curated NEU scratches-versus-inclusion shortcut slice
  when it exists, with synthetic fallback for fresh clones. The real path uses
  a stronger correlated border stripe, balanced train capping, explicit clean
  versus challenge metrics, intervention training that keeps the original
  images plus shortcut-randomised variants, and known-region shortcut
  diagnostics.
- A second real industrial adapter now exists for KolektorSDD2. It writes the
  same shared shortcut manifest contract, so Demo 02 and Demo 08 can also be
  pointed at `data/processed/ksdd2/shortcut_binary/manifest.jsonl`.
- Demo 08 now uses learned industrial classifier drift under blur, contrast,
  compression, lighting, and shadow shifts, and it now switches to the same
  curated NEU shortcut slice for the classifier path when available. It also
  adds an optional local PatchCore anomaly-drift section when MVTec bottle data
  is prepared. When MVTec AD 2 scenario manifests are prepared, the same report
  now adds second-wave anomaly sections for those scenarios. When VisA manifests
  are prepared, it adds cross-dataset anomaly-drift sections there too.
- Demo 07 now adds a category-specific front-label template comparator on local
  MVTec LOCO AD `juice_bottle`, so the report can contrast PatchCore patch
  novelty with a narrow packaging-rule check. Broader category coverage remains
  future work.
- MVTec AD 2 now has local dataset support and optional Demo 08 anomaly-drift
  sections when prepared.
- VisA now has local dataset support and optional Demo 08 anomaly-drift
  sections when prepared.
- Demo 03 now has an explicit pretrained ResNet-18 feature-map path, but it is
  still not an official PatchCore benchmark reproduction. A pretrained
  WideResNet50-2 feature-map option and category-aware report framing are now
  also wired in, and a second local category path is available through capsule.
  Its benchmark panel
  reports local image-level max-patch AUC and top-patch mask diagnostics.
- The suite verifier is now stronger than a pure file-existence check, but it
  is still a local structural-and-semantic smoke test rather than a guarantee
  of benchmark validity or explanation quality.
