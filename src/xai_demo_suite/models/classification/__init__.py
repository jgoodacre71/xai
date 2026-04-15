"""Lightweight classification helpers for shortcut demos."""

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
    "ClassificationResult",
    "HybridShortcutClassifier",
    "ShapeClassifier",
    "StampShortcutClassifier",
    "accuracy",
    "evaluate_classifier",
    "mask_region",
    "predict_label",
    "swap_stamp",
]
