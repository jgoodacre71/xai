# Imported SHAP Notebook Review

## Source

- original path:
  - `/Users/johngoodacre/code/playing-with-time-series/shap.ipynb`
- active notebook copy:
  - `notebooks/global_local_explainability/09_global_vs_local_explainability_shap.ipynb`

## What it is

The original external notebook is an external scratchpad notebook rather than a
clean, single-purpose project notebook.

It contains three distinct strands:

1. a relevant XAI section built around a synthetic hollow-sphere X-ray dataset,
   linear and Ridge baselines, a CNN regressor, and Deep SHAP analysis;
2. a separate phase-aware versus phase-blind signal-analysis section;
3. an unrelated horse-racing and exotic-bet pricing / simulation section.

The original external notebook had 27 cells, used Python 3, and mixed the
relevant XAI material with unrelated later sections.

The active cleaned notebook keeps only the vision/XAI strand and removes the
unrelated material at the end. It now has 19 cells and still has no stored
outputs.

## Relevant section for this repository

The strongest fit with `xai-demo-suite` is the first section:

- synthetic image generation for a hollow metal sphere;
- linear and Ridge regression baselines on pixels;
- a CNN regressor;
- SHAP attribution analysis on the simpler and harder synthetic variants;
- discussion of localisation and attribution behaviour.

This is directionally compatible with the repository's themes:

- evidence maps;
- shortcut-like behaviour under controlled synthetic structure;
- comparison between simple and more expressive models;
- explanation quality under dataset changes.

## Weak fit or out-of-scope content

These sections do not fit the current project shape:

- phase-aware versus phase-blind time-series tasks;
- wavelet and frequency-domain classification content;
- horse-racing pricing, tote flows, and betting simulations.

Those parts are not aligned with the current vision-XAI scope and should not be
treated as candidate demo material for this repository without a separate design
decision.

## What was cleaned

The cleaned copy removes the non-XAI tail from the original notebook:

- phase-aware versus phase-blind signal-analysis content;
- wavelet and time-series classification material;
- horse-racing pricing and betting simulations.

It also adds clearer framing around the real XAI question:

- global versus local explainability;
- why the notebook matters for this repository;
- what was learned;
- residual risks and limitations;
- the most plausible integration direction.

## Technical observations

- The notebook is monolithic and cell-order dependent.
- Repeated imports and repeated model/data sections appear later in the file.
- There is no paired percent-script source.
- Reusable logic currently lives inside the notebook rather than in `src/`.
- The dependency surface is wider than the base repo:
  - `shap`
  - `torch`
  - `scipy`
  - `pywt`
  - `pandas`
- Some code appears to patch or reach into SHAP internals:
  - `import shap.explainers._deep.deep_utils as _du`

That last point is a maintenance risk if this notebook were promoted into the
main demo set.

## Quality after cleanup

After trimming the irrelevant cells, the notebook is materially better:

- the narrative is now coherent from start to finish;
- the simple and hard datasets form a sensible global-versus-local comparison;
- the closing notes make the caveats explicit;
- the remaining content is all within the repository's vision-XAI scope.

Even so, it is still only a **prototype-quality** notebook by this repo's
standards rather than a finished demo notebook.

The main reasons are:

- too much reusable logic still lives in notebook cells;
- the code cells are large and monolithic;
- there is no paired percent-script source for the cleaned derivative;
- there are no tests around the notebook-specific helper logic;
- the dependency surface is wider than the base project;
- SHAP-internal hooks may break over time.

## Recommendation

Do not treat either notebook as a main project notebook as-is.

If you want to reuse it here, the right path is:

1. treat the cleaned copy as the reference prototype;
2. extract the synthetic X-ray plus SHAP section into package code;
3. move dataset generation, model code, and explanation helpers into `src/`;
4. create a clean paired notebook in the repo style;
5. add tests for the reusable logic;
6. decide explicitly whether this becomes:
   - a new demo about regression explanations on synthetic industrial imagery, or
   - a reference/prototype notebook only.

## Suggested next step

If useful, the next practical task is to turn the active notebook
into a true repo-native candidate by splitting it into:

- package code for synthetic image generation and attribution helpers;
- a thin paired notebook;
- a narrow decision on whether this belongs in the curated demo catalogue or
  stays in the reference layer.
