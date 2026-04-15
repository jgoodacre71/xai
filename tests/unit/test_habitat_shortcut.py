from __future__ import annotations

from pathlib import Path

from xai_demo_suite.data.synthetic import (
    generate_habitat_bird_dataset,
    write_habitat_counterfactual,
)
from xai_demo_suite.models.classification import (
    BirdShapeClassifier,
    HabitatShortcutClassifier,
    accuracy,
    evaluate_bird_classifier,
    group_accuracy,
    worst_group_accuracy,
)


def test_habitat_shortcut_fails_crossed_groups_while_shape_classifier_handles_them(
    tmp_path: Path,
) -> None:
    _, test_samples = generate_habitat_bird_dataset(tmp_path)

    habitat_results = evaluate_bird_classifier(HabitatShortcutClassifier(), test_samples)
    shape_results = evaluate_bird_classifier(BirdShapeClassifier(), test_samples)

    assert accuracy(habitat_results) == 0.5
    assert worst_group_accuracy(group_accuracy(test_samples, habitat_results)) == 0.0
    assert accuracy(shape_results) == 1.0
    assert worst_group_accuracy(group_accuracy(test_samples, shape_results)) == 1.0


def test_habitat_counterfactual_rewrites_background_and_flips_shortcut(
    tmp_path: Path,
) -> None:
    _, test_samples = generate_habitat_bird_dataset(tmp_path)
    sample = next(item for item in test_samples if item.sample_id == "test_waterbird_land_000")
    classifier = HabitatShortcutClassifier()

    original = evaluate_bird_classifier(classifier, [sample])[0]
    swapped_path = write_habitat_counterfactual(sample, "water", tmp_path / "swap.png")
    swapped_score = classifier.predict_score(swapped_path)

    assert original.predicted == "landbird"
    assert swapped_path.exists()
    assert swapped_score > 0.0
