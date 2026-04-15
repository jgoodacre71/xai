from __future__ import annotations

from pathlib import Path

from xai_demo_suite.data.synthetic import generate_industrial_shortcut_dataset
from xai_demo_suite.models.classification import (
    ShapeClassifier,
    StampShortcutClassifier,
    accuracy,
    evaluate_classifier,
    mask_region,
    swap_stamp,
)


def test_stamp_shortcut_fails_swapped_cases_while_shape_classifier_handles_them(
    tmp_path: Path,
) -> None:
    _, test_samples = generate_industrial_shortcut_dataset(tmp_path)
    swapped_ids = {"test_normal_swapped_stamp", "test_defect_swapped_stamp"}

    shortcut_results = [
        result
        for result in evaluate_classifier(StampShortcutClassifier(), test_samples)
        if result.sample_id in swapped_ids
    ]
    shape_results = [
        result
        for result in evaluate_classifier(ShapeClassifier(), test_samples)
        if result.sample_id in swapped_ids
    ]

    assert accuracy(shortcut_results) == 0.0
    assert accuracy(shape_results) == 1.0


def test_shortcut_counterfactual_helpers_write_images(tmp_path: Path) -> None:
    _, test_samples = generate_industrial_shortcut_dataset(tmp_path)
    sample = next(item for item in test_samples if item.sample_id == "test_normal_swapped_stamp")

    masked = mask_region(sample.image_path, sample.stamp_region, tmp_path / "masked.png")
    swapped = swap_stamp(sample.image_path, "blue", tmp_path / "swapped.png")

    assert masked.exists()
    assert swapped.exists()
