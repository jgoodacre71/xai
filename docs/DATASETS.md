# Datasets

## Policy

- Raw data is never committed.
- Dataset scripts must preserve source structure where feasible.
- Licences and usage restrictions must be recorded.
- Every dataset adapter must define a canonical processed representation.

## Required first-wave datasets

### MVTec AD
Use for:
- PatchCore baseline
- exemplar retrieval
- nuisance contamination experiments

Source:
- official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- official downloads page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads>

Licence:
- Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0)
- non-commercial use only

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_ad --category bottle --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_ad --category bottle
./.venv/bin/python -m xai_demo_suite.cli.data prepare mvtec_ad --category bottle
```

Storage policy:
- archives are downloaded to `data/raw/mvtec_ad/archives/`;
- extracted copies are written to `data/interim/mvtec_ad/`;
- manifests are written to `data/processed/mvtec_ad/<category>/manifest.jsonl`;
- raw archives, extracted data, and processed manifests are local artefacts and
  are excluded from git.

The full dataset is about 4.9 GB. Prefer one category first while developing the
PatchCore pipeline.

### MVTec LOCO AD
Use for:
- logical anomaly limitations
- structure versus logic comparisons

Source:
- official dataset page:
  <https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad>
- official downloads page:
  <https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad/downloads>

Licence:
- Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0)
- non-commercial use only

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_loco_ad --category juice_bottle --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_loco_ad --category juice_bottle
./.venv/bin/python -m xai_demo_suite.cli.data prepare mvtec_loco_ad --category juice_bottle
```

Storage policy:
- archives are downloaded to `data/raw/mvtec_loco_ad/archives/`;
- extracted copies are written to `data/interim/mvtec_loco_ad/`;
- manifests are written to
  `data/processed/mvtec_loco_ad/<category>/manifest.jsonl`;
- raw archives, extracted data, and processed manifests are local artefacts and
  are excluded from git.

The full dataset is about 5.71 GB. Prefer one category first while developing
the logical-anomaly comparison.

### Waterbirds
Use for:
- classic shortcut demonstration
- background reliance and counterfactual swaps

Source:
- Stanford group DRO repository:
  <https://github.com/kohpangwei/group_DRO>
- dataset tarball:
  <https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz>

Licence / usage:
- the linked Waterbirds tarball does not restate a single unified licence in the
  same way as the MVTec family;
- it is derived from CUB and Places, so this repo treats it conservatively as
  research-only until the upstream terms are checked by the user.

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch waterbirds --category waterbird_complete95_forest2water2 --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data fetch waterbirds --category waterbird_complete95_forest2water2
./.venv/bin/python -m xai_demo_suite.cli.data prepare waterbirds --category waterbird_complete95_forest2water2
```

Storage policy:
- archives are downloaded to `data/raw/waterbirds/archives/`;
- extracted copies are written to `data/interim/waterbirds/`;
- manifests are written to
  `data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl`;
- raw archives, extracted data, and processed manifests are local artefacts and
  are excluded from git.

Current local status:
- when the prepared manifest exists, Demo 01 uses a real Waterbirds report path
  with configurable ResNet-18 tuning, worst-group metrics, Grad-CAM,
  Integrated Gradients, context-masking perturbation checks, and prototype
  exemplars;
- when the manifest is absent, Demo 01 falls back to the generated
  Waterbirds-style proxy under `outputs/waterbirds_shortcut/synthetic/`.

## Strong second-wave datasets

- MVTec AD 2
- MetaShift
- Spawrious
- VisA
- NEU / GC10-DET

### NEU-CLS
Use for:
- real-image industrial shortcut learning
- Demo 02 shortcut correlation checks
- Demo 08 classifier drift with a real industrial data path

Source:
- upstream page:
  <https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm>

Licence / usage:
- the upstream page should be checked before any broader use claim is made;
- this repo treats NEU-CLS conservatively as research-only until the upstream
  terms are confirmed by the user.

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch neu_cls --category shortcut_binary --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data fetch neu_cls --category shortcut_binary --archive-url <direct-archive-url>
./.venv/bin/python -m xai_demo_suite.cli.data prepare neu_cls --category shortcut_binary
```

Storage policy:
- place raw archives under `data/raw/neu_cls/archives/`, or point `prepare` at
  a manual source root with `--source-root`;
- extracted source copies are written to `data/interim/neu_cls/raw/`;
- the prepared binary shortcut layout is written to
  `data/interim/neu_cls/shortcut_binary/`;
- the processed manifest is written to
  `data/processed/neu_cls/shortcut_binary/manifest.jsonl`;
- raw archives, extracted data, prepared copies, and processed manifests are
  local artefacts and are excluded from git.

Notes:
- the current adapter uses a curated scratches-versus-inclusion slice rather
  than all six NEU classes, because that gives Demo 02 and Demo 08 a much
  clearer real-image shortcut-learning story;
- public archives may appear either as the original `IMAGES/<class>_*.bmp`
  layout or as train/valid image splits with names like `crazing_10.jpg`; the
  local preparer now accepts both;
- training images receive a correlated full-height border stripe so the report
  can show the shortcut trap and the intervention on top of real defect
  imagery;
- when the prepared manifest exists, Demo 02 and the classifier section of Demo
  08 use this real-image path automatically.

### MVTec AD 2
Use for:
- second-wave anomaly detection stress tests
- harder acquisition and realism experiments
- future PatchCore robustness evaluation

Source:
- official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2>

Licence:
- Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0)
- non-commercial use only

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_ad_2 --category all --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data fetch mvtec_ad_2 --category all --archive-url <direct-archive-url>
./.venv/bin/python -m xai_demo_suite.cli.data prepare mvtec_ad_2 --category all
```

Notes:
- the official page is currently treated as the authoritative source URL, but
  the repo does not hard-code a brittle direct archive link for fetch;
- prepare discovers scenario folders locally and writes one manifest per
  scenario under `data/processed/mvtec_ad_2/<scenario>/manifest.jsonl`.

### VisA
Use for:
- second-wave industrial anomaly detection
- cross-dataset anomaly stress tests beyond MVTec
- future PatchCore generalisation checks

Source:
- project repository: <https://github.com/amazon-science/spot-diff>
- dataset archive:
  <https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar>
- published one-class split CSV:
  <https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv>

Licence:
- Creative Commons Attribution 4.0 International (CC BY 4.0)

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch visa --category all --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data fetch visa --category all
./.venv/bin/python -m xai_demo_suite.cli.data prepare visa --category all
```

Storage policy:
- the raw archive is downloaded to `data/raw/visa/archives/VisA_20220922.tar`;
- the published split CSV is downloaded to `data/raw/visa/splits/1cls.csv`;
- the extracted raw bundle is written to `data/interim/visa/raw/`;
- the prepared one-class layout is written to `data/interim/visa/1cls/<category>/`;
- manifests are written to `data/processed/visa/<category>/manifest.jsonl`;
- raw archives, split CSV files, extracted data, prepared copies, and processed
  manifests are local artefacts and are excluded from git.

Notes:
- the adapter follows the published one-class split and writes one manifest per
  prepared category;
- prepared VisA manifests now feed optional cross-dataset anomaly-drift
  sections in Demo 08.

### MetaShift
Use for:
- natural-context shortcut learning beyond Waterbirds
- subpopulation shift diagnostics for object/context leakage
- future Pillar A natural-scene extension

Source:
- repository: <https://github.com/Weixin-Liang/MetaShift>
- application docs:
  <https://metashift.readthedocs.io/en/latest/sub_pages/applications.html>

Licence / usage:
- the repository is published under MIT;
- generated image splits depend on upstream Visual Genome / GQA assets and
  should be treated conservatively as research-only until those upstream terms
  are checked by the user.

Local workflow:

```bash
./.venv/bin/python -m xai_demo_suite.cli.data list
./.venv/bin/python -m xai_demo_suite.cli.data fetch metashift --category subpopulation_shift_cat_dog_indoor_outdoor --dry-run
./.venv/bin/python -m xai_demo_suite.cli.data prepare metashift --category subpopulation_shift_cat_dog_indoor_outdoor
```

Storage policy:
- MetaShift is not fetched as a single archive in this repo;
- generate the published `MetaShift-subpopulation-shift` split with the
  upstream scripts and place it under
  `data/external/metashift/MetaShift-subpopulation-shift/`, or pass
  `--source-root` to `prepare`;
- manifests are written to
  `data/processed/metashift/subpopulation_shift_cat_dog_indoor_outdoor/manifest.jsonl`;
- generated image splits and processed manifests are local artefacts and are
  excluded from git.

Notes:
- the current adapter targets the published cat-vs-dog indoor/outdoor
  subpopulation-shift split because it is the most directly relevant natural
  shortcut extension for Demo 01;
- prepared MetaShift manifests now extend Demo 01 with the same group-metric,
  explanation, and perturbation contract used for Waterbirds.

## Synthetic generators to build in-repo

### Nuisance injector
Inject:
- border
- stamp
- habitat background
- corner tab
- lighting gradient
- crop shift
- vignette

### Count generator
Generate repeated objects with controlled missing/extra instances.

### Severity generator
Generate controlled defect intensity levels for showing the severity mismatch problem.

### Logic board generator
Generate slot-based arrangements with valid and invalid configurations.
