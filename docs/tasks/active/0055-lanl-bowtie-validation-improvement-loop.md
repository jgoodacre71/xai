# 0055: LANL Bowtie Validation Improvement Loop

## Status
In progress

## Owner
Codex thread

## Why
The LANL Bowtie PatchCore notebook now has a serious `m1_320_100k` centre-crop
baseline, but further improvement must be driven by validation metrics rather
than by repeatedly looking at test performance. The notebook needs a controlled
improvement harness that can test geometry, location, memory, domain, foreground,
and supervised spatial ideas without violating leakage discipline.

## Source of truth
- `REPO_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/XAI_CONTRACT.md`
- `.agents/skills/patchcore-explainer/SKILL.md`
- `notebooks/01_lanl_bowtie_patchcore_experiment.ipynb`

## Scope
- Preserve the current `m1_320_100k` run stamp as the primary serious baseline.
- Add validation-only experiment registry and improvement-loop controls.
- Add validation-led 100k versus 200k analysis.
- Add serious location-modelling narrative and stage controls.
- Add disk-readiness and safe artefact cleanup reporting.
- Keep all static plots inline and do not save images by default.

## Out of scope
- Automatically running every heavy candidate in one notebook pass.
- Selecting any model by test metrics.
- Changing manifest or split logic.
- Automatic relabelling.
- Automatic deletion of artefacts.

## Deliverables
- Updated LANL Bowtie notebook.
- Persisted CSV/JSON artefacts for registry, stage comparisons, and disk reports.
- Validation run of the current baseline notebook.
- Optional separate execution of heavyweight staged experiments where disk/time allows.

## Constraints
- Memory bank uses accepted training images only.
- Validation selects score type, preprocessing, location method, model family, and threshold.
- Test metrics are frozen audit only.
- Every comparison row must record `manifest_hash`, `split_hash`,
  `uses_test_for_training=False`, `selected_by`, and test cohort counts.
- PatchCore provenance must retain source image ids and patch coordinates.

## Proposed file changes
- `notebooks/01_lanl_bowtie_patchcore_experiment.ipynb` — add controls, registry,
  staged execution support, validation-led comparisons, and cleanup reporting.
- `docs/tasks/active/0055-lanl-bowtie-validation-improvement-loop.md` — durable
  plan for this work.

## Validation plan
1. Parse notebook code cells for syntax errors.
2. Execute the notebook with `BOWTIE_ACTIVE_PRESET=m1_320_100k`.
3. Verify no notebook error outputs.
4. Verify run stamp still reports `m1_320_100k`, `centre_crop`, `score_max`.
5. Verify registry/comparison artefacts include hashes and selected-by fields.
6. Run `git diff --check`.

## Risks
- Heavy letterbox or foreground-box runs may require a new 320 px embedding cache.
- Repeated split or OOF experiments can be expensive; they must remain opt-in.
- Validation AP/operational metrics may not support the same candidate as test AUROC.

## Decision log
### 2026-06-14 12:11:03 BST
- Decision: Keep the notebook default on `m1_320_100k` and make the improvement
  loop opt-in.
- Reason: The current run is the serious primary baseline and should remain the
  visible executed notebook stamp.
- Follow-up: Add one-at-a-time staged commands for letterbox, spatial-bin,
  domain-bank/domain-threshold, OOF supervised, and ensembles.

## Progress log
### 2026-06-14 12:11:03 BST
- Completed: Opened active task.
- Verification: Pending.
- Remaining: Patch notebook, execute baseline, and report outcomes.

### 2026-06-14
- Completed: Added opt-in validation-driven improvement controls to the LANL
  Bowtie notebook; added `experiment_registry.csv`, validation-led 100k versus
  200k analysis, serious location-modelling stage table, disk-readiness report,
  artefact disk-usage report, and explicit 90s target framing.
- Completed: Executed `m1_320_100k` after patching; no notebook error outputs.
- Completed: Executed `m1_320_100k_letterbox` as a separate notebook copy in
  `/tmp/lanl_bowtie_m1_320_100k_letterbox_executed.ipynb`; no notebook error
  outputs.
- Verification: `git diff --check` passed.
- Result: Validation did not support letterbox. Best letterbox row was
  `score_topk_mean_fg` with validation AUROC `0.748766` and AP `0.240476`,
  below the current centre-crop 100k baseline (`score_max`, validation AUROC
  `0.803159`, AP `0.352075`) and the cached 200k validation-selected challenger
  (`score_max_fg`, validation AUROC `0.809977`, AP `0.359232`).
- Completed: Added environment overrides for opt-in location-bin, domain,
  supervised spatial, and rank-ensemble candidate runs; added registry ingestion
  for spatial-bin, location-aware, supervised spatial, domain-threshold, and
  rank-ensemble reports.
- Completed: Executed spatial-bin PatchCore candidates on cached `m1_320_100k`
  embeddings. Validation did not support promotion. Best retained spatial-bin
  rows were below the primary centre-crop baseline by validation AUROC.
- Completed: Executed OOF supervised spatial meta-models using out-of-fold
  PatchCore train maps. Best validation AUROC was `0.830943` for
  `HistGradientBoosting leaves=15 lr=0.06`; best validation AP/F1 came from
  ExtraTrees variants but still did not reach the 0.90 AUROC target.
- Completed: Added and executed validation-selected rank-average ensembles over
  PatchCore scores and OOF-supervised spatial model scores. Current validation
  best is `Rank ensemble: score_max_fg + HistGradientBoosting leaves=15 lr=0.06`
  with validation AUROC `0.835474`, validation AP `0.401701`; frozen test audit
  for that validation-selected row is AUROC `0.864296`, AP `0.550269`,
  precision `0.326203`, recall `0.7625`, FPR `0.186667`, F1 `0.456929`.
- Completed: Replaced split-local rank normalisation with a deployable
  validation-fitted empirical-CDF ensemble transform. Component transforms are
  fitted on validation scores only and then frozen for test/live scoring.
- Completed: Added paired baseline-versus-challenger reports:
  `baseline_vs_challenger_summary.csv`,
  `baseline_challenger_bootstrap_deltas.csv`, `challenger_val_scores.csv`,
  `challenger_test_scores.csv`, HGB permutation-importance reports, and
  `final_decision_table.csv`.
- Result: The validation-ECDF ensemble challenger improves validation AUROC/AP
  from `0.803159`/`0.352075` to `0.835485`/`0.401499`. Frozen test audit is
  mixed: AUROC drops from `0.877426` to `0.864306`, AP rises from `0.513390`
  to `0.545750`, precision@top10 rises from `0.486842` to `0.552632`, and
  recall@FPR5 rises from `0.4375` to `0.5250`.
- Result: Paired group-bootstrap deltas support keeping the ECDF ensemble as a
  triage challenger, not a replacement baseline. With 1000 paired group-bootstrap
  replicates, test delta AUROC is `-0.013120` with 95% CI
  `[-0.046511, 0.016912]`; test delta AP is `0.032360` with 95% CI
  `[-0.041200, 0.104112]`; test delta precision@top10 is `0.065789` with
  95% CI `[-0.013333, 0.142857]`; test delta recall@FPR5 is `0.087500` with
  95% CI `[-0.011523, 0.214305]`; test delta FPR@recall70 is `0.026667`
  with 95% CI `[-0.035934, 0.079368]`, where negative favours the challenger.
- Decision: Keep `m1_320_100k centre_crop score_max` as the primary PatchCore
  baseline. Retain the validation-ECDF ensemble as a leakage-safe triage
  challenger for inspection-budget workflows.
- Completed: Added explicit notebook roles (`primary_baseline` and
  `triage_challenger`), a use-case recommendation table, validation-only ECDF
  sanity checks, promotion criteria, detailed disagreement review panels,
  disagreement domain summaries, HGB permutation-importance interpretation, and
  a label/defect review queue with controlled review statuses.
- Completed: Re-ran `m1_320_100k_letterbox` after disk cleanup. Validation still
  does not support letterbox: best validation row was `score_topk_mean_fg` with
  validation AUROC `0.748766` and AP `0.240476`, so no letterbox ECDF ensemble
  was run.
- Completed: Updated comparison rows to include `manifest_hash`, `split_hash`,
  `test_n`, `test_rejects`, `test_accepts`, `selected_by=validation`,
  `uses_test_for_training=False`, and `strictly_apples_to_apples=True`.
- Completed: Refreshed the default `m1_320_100k` notebook in place after the
  candidate runs. The visible run stamp remains the primary PatchCore baseline:
  `score_max`, test AUROC `0.877426`, AP `0.513390`, F1 `0.502058`.
- Verification: final notebook has no error outputs; opt-in ECDF challenger
  execution completed to `/tmp/lanl_bowtie_m1_320_100k_ecdf_ensemble_executed.ipynb`;
  default `m1_320_100k` execution completed in place; `git diff --check`
  passed.
- Remaining: The validation loop has improved validation AUROC from `0.803` to
  `0.835`, but not into the 90s. Next serious stages are foreground-box
  registration, domain-specific memory banks, geometry-preserving location
  experiments on letterbox/registered crops, more explicit defect/label audit,
  and possibly a locked repeated-group-split estimate before any production
  claim.
