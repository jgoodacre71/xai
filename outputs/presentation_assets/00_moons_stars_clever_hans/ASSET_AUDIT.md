# Demo 00 Asset Audit

Target folder: `outputs/presentation_assets/00_moons_stars_clever_hans/`

This audit records dimensions, file size, inferred purpose, presentation worthiness, recommended slide use, and keep/improve/discard decisions.

## Primary Assets

| filename | type | dimensions | size | description | presentation-worthy | recommended use | decision |
|---|---|---:|---:|---|---|---|---|
| `01_apparent_task_shape_examples.png` | PNG | 1532 x 1022 | 154.1 KB | Apparent moon/star shape task examples. | Yes | Slide 2: apparent task | keep |
| `02_both_models_perfect_iid.png` | PNG | 1780 x 770 | 89.7 KB | Result card showing both models appear perfect on IID validation. | Yes | Slide 3: apparent success | keep |
| `03_many_functions_pass_same_exam.png` | PNG | 2004 x 1024 | 123.3 KB | Central non-identifiability slide: many functions pass the same exam. | Yes | Slide 3: thesis | keep |
| `04_same_shape_movement_counterfactual.png` | PNG | 1343 x 1084 | 227.5 KB | Four-panel counterfactual: same shape moved to shortcut-breaking position. | Yes | Slide 4: Act I reveal | keep |
| `05_moon_movement_confidence_path.png` | PNG | 1808 x 1055 | 143.2 KB | Moon movement confidence path with MLP flip. | Yes | Slide 4: moon movement evidence | keep |
| `06_star_movement_confidence_path.png` | PNG | 1808 x 1055 | 147.6 KB | Star movement confidence path with MLP flip. | Yes | Slide 4 backup or appendix | keep |
| `07_position_data_audit_scatter_histogram.png` | PNG | 2065 x 726 | 259.7 KB | Object-centre scatter and x-y histogram: address encodes label. | Yes | Slide 5: position data audit | keep |
| `08_position_rule_and_nearest_neighbours.png` | PNG | 1920 x 716 | 209.6 KB | Position-only rule and nearest-neighbour demo. | Yes | Slide 5: silly shortcut model | keep |
| `09_shape_morph_strip.png` | PNG | 1715 x 1304 | 201.6 KB | Shape morph strip; useful support but not central to the 12-slide story. | Maybe, with cropping/text enlargement | Appendix or backup | improve |
| `10_position_response_maps.png` | PNG | 1836 x 712 | 75.9 KB | Response maps showing where models believe stars live. | Yes | Slide 6: response geometry | keep |
| `11_shape_position_surface.png` | PNG | 1611 x 912 | 840.0 KB | Shape-position score surface; strong geometry support, text may be dense. | Maybe, with cropping/text enlargement | Slide 6 backup | improve |
| `12_act2_apparent_background_examples.png` | PNG | 1836 x 712 | 61.6 KB | Act II examples: backgrounds look the same. | Yes | Slide 7: Act II setup | keep |
| `13_background_only_sanity_check.png` | PNG | 1900 x 1420 | 373.5 KB | Background-only sanity check: hidden cue alone solves biased exam. | Yes | Slide 7 support | keep |
| `14_invisible_background_swap_counterfactual.png` | PNG | 1360 x 786 | 84.7 KB | Act II killer counterfactual: invisible background shift flips belief. | Yes | Slide 8: Act II reveal | keep |
| `15_background_confidence_sweep.png` | PNG | 1752 x 912 | 112.1 KB | Confidence sweep across invisible background interpolation. | Yes | Slide 8 support | keep |
| `16_amplified_background_cue_and_histogram.png` | PNG | 1247 x 520 | 49.2 KB | Amplified background cue reveal plus red-channel histogram. | Yes | Slide 9: cue reveal | keep |
| `17_background_cue_snr_card.png` | PNG | 2236 x 698 | 323.3 KB | SNR card: invisible per pixel but strong in aggregate. | Yes | Slide 9: statistical explanation | keep |
| `18_background_data_audit_histograms.png` | PNG | 1904 x 716 | 277.5 KB | Background-statistics audit: red-channel mean separates labels. | Yes | Slide 9: data audit | keep |
| `19_background_rule_and_nearest_neighbours.png` | PNG | 1836 x 641 | 126.2 KB | Background-only rule and nearest-neighbour demo. | Yes | Slide 9 or backup | keep |
| `20_heatmaps_are_not_enough.png` | PNG | 2004 x 941 | 67.0 KB | Compact saliency caution: heatmaps are supporting evidence only. | Yes | Slide 10: heatmaps caution | keep |
| `21_act2_mitigation_retest.png` | PNG | 1948 x 744 | 135.8 KB | Mitigation/re-test comparison: biased versus mitigated CNN. | Yes | Slide 11: intervention and re-test | keep |
| `22_xai_factor_control_loop.png` | PNG | 1614 x 548 | 67.0 KB | Final loop: observe, hypothesise, intervene, re-test, monitor. | Yes | Slide 12: XAI loop | keep |
| `23_counterfactual_minimum_change_cards.png` | PNG | 2141 x 798 | 121.2 KB | Counterfactual minimum-change cards; good closing support. | Yes | Slide 12 support | keep |
| `anim_01_moon_moves_confidence.gif` | GIF | 1040 x 600 | 255.0 KB | Hero animation: same moon moves and MLP flips. | Yes | Slide 4: play on click | keep |
| `anim_02_star_moves_confidence.gif` | GIF | 1040 x 600 | 391.8 KB | Hero animation: same star moves and MLP flips. | Yes | Slide 4 optional second reveal | keep |
| `anim_03_response_map_path_mlp.gif` | GIF | 1288 x 560 | 371.7 KB | Moving object across MLP territory map. | Yes | Slide 6: play on click | keep |
| `anim_04_response_map_path_cnn.gif` | GIF | 1288 x 560 | 870.6 KB | Moving object across CNN response map; model follows shape more. | Yes | Slide 6 backup | keep |
| `anim_05_moon_moves_heatmaps.gif` | GIF | 1708 x 840 | 421.6 KB | Heatmap movement animation; useful only as caution/appendix. | No, appendix only | Appendix only | discard |
| `anim_06_star_moves_heatmaps.gif` | GIF | 1708 x 840 | 383.7 KB | Heatmap movement animation; useful only as caution/appendix. | No, appendix only | Appendix only | discard |
| `anim_07_morph_lower_left.gif` | GIF | 900 x 520 | 109.1 KB | Morph animation at lower-left; support for shape-position story. | Maybe, with cropping/text enlargement | Backup/appendix | improve |
| `anim_08_morph_upper_right.gif` | GIF | 900 x 520 | 107.3 KB | Morph animation at upper-right; support for shape-position story. | Maybe, with cropping/text enlargement | Backup/appendix | improve |
| `anim_09_morph_lower_left_heatmaps.gif` | GIF | 1568 x 756 | 298.2 KB | Morph heatmap animation; too secondary for main deck. | No, appendix only | Discard from Google pack | discard |
| `anim_10_morph_upper_right_heatmaps.gif` | GIF | 1568 x 756 | 260.7 KB | Morph heatmap animation; too secondary for main deck. | No, appendix only | Discard from Google pack | discard |
| `anim_11_invisible_background_morph_moon.gif` | GIF | 980 x 560 | 1.1 MB | Hero animation: same moon, invisible background morph, CNN flips. | Yes | Slide 8: play on click | keep |
| `anim_12_invisible_background_morph_star.gif` | GIF | 980 x 560 | 2.3 MB | Hero animation: same star, invisible background morph, CNN flips. | Yes | Slide 8 optional second reveal | keep |
| `anim_01_moon_moves_confidence.mp4` | MP4 | 1040 x 600 | 49.8 KB | Hero animation: same moon moves and MLP flips. | Yes | Slide 4: play on click | keep |
| `anim_02_star_moves_confidence.mp4` | MP4 | 1040 x 600 | 48.0 KB | Hero animation: same star moves and MLP flips. | Yes | Slide 4 optional second reveal | keep |
| `anim_03_response_map_path_mlp.mp4` | MP4 | 1288 x 560 | 67.6 KB | Moving object across MLP territory map. | Yes | Slide 6: play on click | keep |
| `anim_04_response_map_path_cnn.mp4` | MP4 | 1288 x 560 | 63.4 KB | Moving object across CNN response map; model follows shape more. | Yes | Slide 6 backup | keep |
| `anim_05_moon_moves_heatmaps.mp4` | MP4 | 1708 x 840 | 94.2 KB | Heatmap movement animation; useful only as caution/appendix. | No, appendix only | Appendix only | discard |
| `anim_06_star_moves_heatmaps.mp4` | MP4 | 1708 x 840 | 99.7 KB | Heatmap movement animation; useful only as caution/appendix. | No, appendix only | Appendix only | discard |
| `anim_07_morph_lower_left.mp4` | MP4 | 900 x 520 | 35.7 KB | Morph animation at lower-left; support for shape-position story. | Maybe, with cropping/text enlargement | Backup/appendix | improve |
| `anim_08_morph_upper_right.mp4` | MP4 | 900 x 520 | 36.0 KB | Morph animation at upper-right; support for shape-position story. | Maybe, with cropping/text enlargement | Backup/appendix | improve |
| `anim_09_morph_lower_left_heatmaps.mp4` | MP4 | 1568 x 756 | 67.5 KB | Morph heatmap animation; too secondary for main deck. | No, appendix only | Discard from Google pack | discard |
| `anim_10_morph_upper_right_heatmaps.mp4` | MP4 | 1568 x 756 | 62.7 KB | Morph heatmap animation; too secondary for main deck. | No, appendix only | Discard from Google pack | discard |
| `anim_11_invisible_background_morph_moon.mp4` | MP4 | 980 x 560 | 35.6 KB | Hero animation: same moon, invisible background morph, CNN flips. | Yes | Slide 8: play on click | keep |
| `anim_12_invisible_background_morph_star.mp4` | MP4 | 980 x 560 | 33.9 KB | Hero animation: same star, invisible background morph, CNN flips. | Yes | Slide 8 optional second reveal | keep |

## Review And Instruction Files

| filename | type | dimensions | size | description | presentation-worthy | recommended use | decision |
|---|---|---:|---:|---|---|---|---|
| `README.md` | MD | n/a | 4.0 KB | Folder manifest or instructions. | Yes | Reference | keep |
| `contact_sheet_pngs.jpg` | JPG | 1384 x 2822 | 573.4 KB | Generated support asset. | Maybe, with cropping/text enlargement | Review manually | improve |
| `contact_sheet_gifs_first_frames.jpg` | JPG | 1568 x 1544 | 288.8 KB | Generated support asset. | Maybe, with cropping/text enlargement | Review manually | improve |
| `contact_sheet_gifs_motion_strip.jpg` | JPG | 1588 x 2746 | 626.7 KB | Generated support asset. | Maybe, with cropping/text enlargement | Review manually | improve |

## Generated Review Pack Files

| filename | type | dimensions | size | purpose |
|---|---|---:|---:|---|
| `README.md` | MD | n/a | 4.0 KB | Original exported asset manifest. |
| `README_GOOGLE_SLIDES.md` | MD | n/a | 1.4 KB | Google Slides upload and playback instructions. |
| `SLIDE_MANIFEST.md` | MD | n/a | 5.8 KB | 12-slide assembly plan with notes and playback guidance. |
| `ASSET_QUALITY_REPORT.md` | MD | n/a | 2.9 KB | Curated quality report: best, backup, and weak assets. |
| `ASSET_AUDIT.md` | MD | n/a | 9.3 KB | This complete asset audit. |
| `asset_pack_summary.json` | JSON | n/a | 986 B | Machine-readable generation summary. |
| `contact_sheet_pngs.jpg` | JPG | 1384 x 2822 | 573.4 KB | Review sheet for all PNGs. |
| `contact_sheet_gifs_first_frames.jpg` | JPG | 1568 x 1544 | 288.8 KB | Review sheet for GIF first frames. |
| `contact_sheet_gifs_motion_strip.jpg` | JPG | 1588 x 2746 | 626.7 KB | Review sheet showing GIF motion strips. |
