# XAI Demo Slide Storyboard

## Design principle

This is not an image gallery.

Each slide should have:
- one claim-style title;
- one mathematical or experimental point;
- one main proof object;
- one short interpretation sentence;
- longer explanation in speaker notes.

Avoid titles like "Act I" or "Act II" before the reveal. The story should unfold.

---

## Slide 1 — Accuracy is not understanding

Claim:
A model can pass the exam while learning the wrong function.

Visual:
`assets/images/slide_plates/plate_12_final_synthesis.png`

Speaker note:
Open with the thesis: IID accuracy does not identify the human concept.

---

## Slide 2 — What supervised learning optimises

Claim:
Training minimises an empirical risk functional.

Equation:
`assets/equations/svg/erm_functional.svg`
`assets/equations/svg/training_objective.svg`

Visual:
`assets/images/slide_plates/plate_01_supervised_problem.png`

Speaker note:
Define the ordinary supervised-learning setup before any XAI.

---

## Slide 3 — Many functions can pass the same finite exam

Claim:
Low empirical risk does not uniquely identify the semantic rule.

Equation:
`assets/equations/svg/many_functions_same_exam.svg`

Visual:
`assets/images/slide_plates/plate_02_erm_many_functions.png`

Speaker note:
This is the non-identifiability pivot.

---

## Slide 4 — Same object. Different position. Different answer.

Claim:
Changing only position changes the MLP belief.

Equation:
`assets/equations/svg/position_counterfactual.svg`

Visual:
`assets/images/slide_plates/plate_04_position_counterfactual.png`

Animation slot:
`assets/video/mp4/anim_moon_moves_confidence.mp4`

Fallback:
`assets/video/first_frames/anim_moon_moves_confidence_first_frame.png`

Speaker note:
Do not explain the data bias yet. Let the counterfactual surprise land.

---

## Slide 5 — The score lives on a factor space

Claim:
XAI asks which factor changes the score.

Equation:
`assets/equations/svg/factorised_generator.svg`
`assets/equations/svg/nuisance_orbit.svg`

Visual:
`assets/images/slide_plates/plate_05_response_map_geometry.png`

Speaker note:
Introduce factor-control language.

---

## Slide 6 — Response maps reveal learned geometry

Claim:
The MLP learned a position boundary; the CNN is stable under this position probe.

Equation:
`assets/equations/svg/response_map.svg`
`assets/equations/svg/decision_contour.svg`

Visual:
`assets/images/slide_plates/plate_05_response_map_geometry.png`

Speaker note:
Frame maps as score geometry, not just heatmaps. The current notebook source provides a static response-map proof object rather than a response-map path animation.

---

## Slide 7 — The shortcut was already in the data

Claim:
A rule that ignores shape can pass the biased exam.

Visual:
`assets/images/slide_plates/plate_06_position_data_audit.png`

Speaker note:
The model did not cheat. The dataset rewarded the shortcut.

---

## Slide 8 — The convolutional model appears to work again

Claim:
A stronger-looking model can still pass IID validation while relying on another cue.

Visual:
`assets/images/slide_plates/plate_07_cnn_second_comfort.png`

Speaker note:
This is the second false comfort.

---

## Slide 9 — Same object. Invisible background shift. Different belief.

Claim:
Changing only a tiny background statistic flips the CNN.

Visual:
`assets/images/slide_plates/plate_08_background_counterfactual.png`

Animation slot:
`assets/video/mp4/anim_invisible_background_morph_moon.mp4`

Fallback:
`assets/video/first_frames/anim_invisible_background_morph_moon_first_frame.png`

Speaker note:
The object is fixed. The background statistic changes.

---

## Slide 10 — Human invisibility is not statistical irrelevance

Claim:
A tiny per-pixel cue can become decisive after aggregation.

Equation:
`assets/equations/svg/background_snr.svg`

Visual:
`assets/images/slide_plates/plate_10_background_data_audit.png`

Speaker note:
This explains why a visually subtle cue can dominate.

---

## Slide 11 — Heatmaps are supporting evidence, not the explanation

Claim:
A plausible heatmap does not identify the controlling factor.

Equation:
`assets/equations/svg/smoothgrad.svg`

Visual:
`assets/images/slide_plates/plate_09_xai_heatmap_caution.png`

Speaker note:
The counterfactual identifies control; the heatmap is supporting evidence.

---

## Slide 12 — Explanation must lead to intervention and re-test

Claim:
The fix is process change plus the same behavioural probe again.

Equation:
`assets/equations/svg/intervention_loop.svg`
`assets/equations/svg/orbit_averaged_risk.svg`

Visual:
`assets/images/slide_plates/plate_11_mitigation_retest.png`

Speaker note:
Do not trust architecture alone; change the data process and re-test.

---

## Slide 13 — What the data taught the model

Claim:
Both failures were available in the data distribution.

Visual:
`assets/images/slide_plates/plate_06_position_data_audit.png`
`assets/images/slide_plates/plate_10_background_data_audit.png`

Speaker note:
This consolidates both shortcuts.

---

## Slide 14 — XAI as experimental model science

Claim:
XAI is not only audit; it is how we make learned structure visible, testable, and correctable.

Equation:
`assets/equations/svg/intervention_loop.svg`

Visual:
`assets/images/slide_plates/plate_12_final_synthesis.png`

Speaker note:
Close by tying the demo to broader research: anomaly detection, scientific ML, monitoring, failure discovery.
