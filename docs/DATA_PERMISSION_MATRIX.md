# Data permission matrix

This matrix is a planning and replication aid, not legal advice. When terms are
unclear, use the conservative status `verify before work use`.

| Dataset | Demo role | Expected local manifest path | Official source | Licence / terms | Work use status | Approximate size | Local home status | Work status | Required approval | Citation requirement | Notes and risks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generated Moons/Stars controlled demo | No-permission Clever-Hans opener | Not applicable | Generated in notebook | Repo-authored generated data | Allowed | Tiny | Available by running notebook | Available by running notebook | None expected | Cite repo/demo if reused | Didactic only; not real-world proof |
| Waterbirds | Natural shortcut benchmark | `data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl` | Stanford group DRO repository and linked tarball | Verify upstream CUB and Places terms before use | Verify before work use | Not recorded in repo | Present when local manifest exists | Unknown | Workplace data approval recommended | Sagawa et al.; upstream dataset citations | Derived dataset; do not redistribute without checking upstream terms |
| NEU-CLS shortcut binary split | Real industrial side-band shortcut and classifier drift | `data/processed/neu_cls/shortcut_binary/manifest.jsonl` | Northeastern University surface defect page | Verify official NEU-CLS terms before use | Verify before work use | Small image dataset; exact archive varies | Present when local manifest exists | Unknown | Workplace data approval recommended | Song and Yan / official source citation | Verify redistribution and publication terms before workplace or external use |
| MVTec AD | PatchCore anomaly provenance | `data/processed/mvtec_ad/<category>/manifest.jsonl` | MVTec AD official page | CC BY-NC-SA 4.0, non-commercial | Non-commercial only; work use requires approval | About 4.9 GB full dataset | Optional local prepared categories | Unknown | Required for commercial/work setting | Bergmann et al. | Strong benchmark, but licence restricts commercial use |
| MVTec LOCO AD | PatchCore logical anomaly limits | `data/processed/mvtec_loco_ad/<category>/manifest.jsonl` | MVTec LOCO AD official page | CC BY-NC-SA 4.0, non-commercial | Non-commercial only; work use requires approval | About 5.71 GB full dataset | Optional local prepared categories | Unknown | Required for commercial/work setting | Bergmann et al. | Useful for logic limits; do not hard-code as only option |
| MVTec AD 2 | Harder anomaly and drift scenarios | `data/processed/mvtec_ad_2/<scenario>/manifest.jsonl` | MVTec AD 2 official page | CC BY-NC-SA 4.0, non-commercial | Non-commercial only; work use requires approval | Verify from official release | Optional local prepared scenarios | Unknown | Required for commercial/work setting | MVTec AD 2 citation | Use only if terms are acceptable at work |
| VisA | More permissive anomaly alternative | `data/processed/visa/<category>/manifest.jsonl` | AWS Open Data / Amazon Science Spot-the-Difference | CC BY 4.0 | Likely permitted with attribution; verify internally | Listed as 10,821 images with image- and pixel-level annotations | Optional local prepared categories | Unknown | Internal approval recommended | Zou et al. | Good candidate when MVTec non-commercial terms are unsuitable |
| KolektorSDD2 | Backup industrial shortcut/anomaly source | `data/processed/ksdd2/shortcut_binary/manifest.jsonl` | VICOS KolektorSDD2 page | CC BY-NC-SA 4.0, non-commercial | Non-commercial only; work use requires approval | Verify from official release | Optional local prepared manifest | Unknown | Required for commercial/work setting | Tabernik et al. | Useful backup, but terms may block work use |
| IEEE candidate datasets | Candidate pool for approved future demos | Not selected yet | IEEE DataPort | Open-access, standard, competition, or unknown depending on dataset | Verify before work use | Dataset-specific | Not selected | Unknown | Required before download/use | IEEE DataPort attribution/citation required | Standard datasets may require subscriber access; do not select prematurely |

## Cautious defaults

- Do not commit raw datasets, archives, extracted images, or processed manifests.
- Treat unclear terms as `verify before work use`.
- Treat non-commercial licences as blocked for commercial/work use until the
  organisation approves the use case.
- Record citations before a dataset becomes part of a public demo.
