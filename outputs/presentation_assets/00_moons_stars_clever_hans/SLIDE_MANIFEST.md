# Demo 00 Slide Manifest

A 12-slide Google Slides assembly plan built around the five strongest moments, not a full asset gallery.

## Slide 1: Moons, Stars, and Clever Hans

**Message:** Accuracy can say the model passed while hiding which function it learned.

**Assets:**
- `03_many_functions_pass_same_exam.png`

**Animation playback:** No animation.

**Speaker note:** Open with the thesis: high IID accuracy does not identify the human concept.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 2: The apparent task

**Message:** This looks like ordinary shape recognition: moon versus star.

**Assets:**
- `01_apparent_task_shape_examples.png`

**Animation playback:** No animation.

**Speaker note:** Do not mention position or background shortcuts yet.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 3: Both models appear perfect

**Message:** IID validation says both models solved the exam, but many functions can pass that exam.

**Assets:**
- `02_both_models_perfect_iid.png`
- `03_many_functions_pass_same_exam.png`

**Animation playback:** No animation.

**Speaker note:** Use this as the non-identifiability pivot before the reveal.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 4: Act I reveal: same moon, different place

**Message:** Nothing semantic changes, but the MLP changes its answer when the object moves.

**Assets:**
- `04_same_shape_movement_counterfactual.png`
- `anim_01_moon_moves_confidence.mp4`
- `05_moon_movement_confidence_path.png`

**Animation playback:** Play the moon movement animation on click.

**Speaker note:** This is the first emotional reveal. Keep the moon animation central; use the path as the quantitative backup.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 5: The position shortcut was already in the data

**Message:** A rule that never looks at shape can pass the biased exam.

**Assets:**
- `07_position_data_audit_scatter_histogram.png`
- `08_position_rule_and_nearest_neighbours.png`

**Animation playback:** No animation.

**Speaker note:** Emphasise that the model did not cheat; the training distribution leaked the answer.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 6: Response geometry: the MLP learned territory

**Message:** Response maps show the MLP score follows address, while the CNN is less tied to territory.

**Assets:**
- `10_position_response_maps.png`
- `anim_03_response_map_path_mlp.mp4`
- `anim_04_response_map_path_cnn.mp4`

**Animation playback:** Play response-map animations on click.

**Speaker note:** Frame these as low-dimensional slices through the learned score function.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 7: Act II: the CNN appears to rescue us

**Message:** The task still looks like shape recognition, and the CNN again appears to work.

**Assets:**
- `12_act2_apparent_background_examples.png`
- `13_background_only_sanity_check.png`

**Animation playback:** No animation.

**Speaker note:** Use this to fool the audience a second time.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 8: Act II reveal: same shape, invisible background shift

**Message:** The object stays fixed, but an almost invisible background cue flips the CNN.

**Assets:**
- `14_invisible_background_swap_counterfactual.png`
- `anim_11_invisible_background_morph_moon.mp4`
- `15_background_confidence_sweep.png`

**Animation playback:** Play the invisible-background morph on click.

**Speaker note:** This is the second emotional reveal. The moon background animation is the hero.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 9: Human invisibility is not statistical irrelevance

**Message:** The background cue is weak per pixel but linearly separable after aggregation.

**Assets:**
- `16_amplified_background_cue_and_histogram.png`
- `17_background_cue_snr_card.png`
- `18_background_data_audit_histograms.png`
- `19_background_rule_and_nearest_neighbours.png`

**Animation playback:** No animation.

**Speaker note:** Show that a silly background-only rule and nearest-neighbour model also pass the biased exam.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 10: XAI is not heatmaps

**Message:** Heatmaps can look plausible while counterfactuals reveal what controls the score.

**Assets:**
- `20_heatmaps_are_not_enough.png`

**Animation playback:** No animation.

**Speaker note:** Keep this short. Saliency is a caveat, not the explanation.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 11: Intervention and re-test

**Message:** The fix is changing the data/process and re-testing behaviour, not producing a nicer heatmap.

**Assets:**
- `21_act2_mitigation_retest.png`
- `23_counterfactual_minimum_change_cards.png`

**Animation playback:** No animation.

**Speaker note:** Make the maturity point: detect shortcut, intervene, re-test the same probe.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.

## Slide 12: XAI as experimental model science

**Message:** The useful explanation is the factor that controls the model score under intervention.

**Assets:**
- `22_xai_factor_control_loop.png`

**Animation playback:** No animation.

**Speaker note:** Close with: the model did not cheat; many functions solved the exam; XAI revealed which one was learned.

**Needs rebuilding:** None required; crop or enlarge only if the Google Slides layout needs it.
