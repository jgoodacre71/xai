"""Lightweight classification helpers for shortcut demos."""

from xai_demo_suite.models.classification.habitat_shortcut import (
    BirdShapeClassifier,
    GroupMetric,
    HabitatShortcutClassifier,
    evaluate_bird_classifier,
    group_accuracy,
    predict_bird_label,
    worst_group_accuracy,
)
from xai_demo_suite.models.classification.shortcut import (
    ClassificationResult,
    HybridShortcutClassifier,
    ShapeClassifier,
    StampShortcutClassifier,
    accuracy,
    evaluate_classifier,
    mask_region,
    predict_label,
    swap_stamp,
)

__all__ = [
    "BirdShapeClassifier",
    "ClassificationResult",
    "GroupMetric",
    "HabitatShortcutClassifier",
    "HybridShortcutClassifier",
    "ShapeClassifier",
    "StampShortcutClassifier",
    "accuracy",
    "evaluate_bird_classifier",
    "evaluate_classifier",
    "group_accuracy",
    "mask_region",
    "predict_bird_label",
    "predict_label",
    "swap_stamp",
    "worst_group_accuracy",
]
