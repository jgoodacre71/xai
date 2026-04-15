"""Lightweight classification helpers for shortcut demos."""

from xai_demo_suite.models.classification.shortcut import (
    ClassificationResult,
    ShapeClassifier,
    StampShortcutClassifier,
    accuracy,
    evaluate_classifier,
    mask_region,
    swap_stamp,
)

__all__ = [
    "ClassificationResult",
    "ShapeClassifier",
    "StampShortcutClassifier",
    "accuracy",
    "evaluate_classifier",
    "mask_region",
    "swap_stamp",
]
