"""Deterministic classifiers for the synthetic Waterbirds-style demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from xai_demo_suite.data.synthetic import HabitatBirdSample
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.models.classification.shortcut import ClassificationResult


class HabitatBirdClassifier(Protocol):
    """Protocol for deterministic habitat shortcut classifiers."""

    @property
    def name(self) -> str:
        """Stable classifier name used in reports."""
        ...

    def predict_score(self, image_path: Path) -> float:
        """Return a positive score for the waterbird class."""

    @property
    def evidence_region(self) -> BoundingBox:
        """Return the primary evidence region used by the classifier."""


@dataclass(frozen=True, slots=True)
class GroupMetric:
    """Accuracy for one label/habitat group."""

    group: str
    accuracy: float
    count: int


def _image_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _region_array(image_path: Path, box: BoundingBox) -> np.ndarray:
    rgb = _image_array(image_path)
    return rgb[box.y : box.y + box.height, box.x : box.x + box.width, :]


@dataclass(frozen=True, slots=True)
class HabitatShortcutClassifier:
    """Classifier that intentionally predicts from background habitat."""

    name: str = "habitat_shortcut"
    habitat_region: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=0, y=0, width=160, height=120)
    )

    def predict_score(self, image_path: Path) -> float:
        region = _region_array(image_path, self.habitat_region)
        blue = float(region[:, :, 2].mean())
        green = float(region[:, :, 1].mean())
        return blue - green

    @property
    def evidence_region(self) -> BoundingBox:
        """Return the habitat evidence region."""

        return self.habitat_region


@dataclass(frozen=True, slots=True)
class BirdShapeClassifier:
    """Classifier that uses the bird silhouette rather than the habitat."""

    name: str = "bird_shape_intervention"
    bird_region: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=24, y=24, width=112, height=84)
    )
    right_beak_region: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=116, y=28, width=24, height=20)
    )
    left_beak_region: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=20, y=34, width=25, height=24)
    )

    def predict_score(self, image_path: Path) -> float:
        right_beak = _region_array(image_path, self.right_beak_region)
        left_beak = _region_array(image_path, self.left_beak_region)
        right_score = _orange_score(right_beak)
        left_score = _orange_score(left_beak)
        return right_score - left_score

    @property
    def evidence_region(self) -> BoundingBox:
        """Return the bird silhouette evidence region."""

        return self.bird_region


def _orange_score(region: np.ndarray) -> float:
    red = region[:, :, 0]
    green = region[:, :, 1]
    blue = region[:, :, 2]
    return float((red * 1.2 + green * 0.6 - blue * 1.4).mean())


def predict_bird_label(score: float) -> str:
    """Convert a waterbird score into a class label."""

    return "waterbird" if score > 0.0 else "landbird"


def evaluate_bird_classifier(
    classifier: HabitatBirdClassifier,
    samples: list[HabitatBirdSample],
) -> list[ClassificationResult]:
    """Evaluate a classifier on synthetic habitat-bird samples."""

    results: list[ClassificationResult] = []
    for sample in samples:
        score = classifier.predict_score(sample.image_path)
        predicted = predict_bird_label(score)
        results.append(
            ClassificationResult(
                sample_id=sample.sample_id,
                label=sample.label,
                predicted=predicted,
                score=score,
                correct=predicted == sample.label,
            )
        )
    return results


def group_accuracy(
    samples: list[HabitatBirdSample],
    results: list[ClassificationResult],
) -> tuple[GroupMetric, ...]:
    """Return per-group accuracies for label and habitat combinations."""

    results_by_id = {result.sample_id: result for result in results}
    groups = sorted({sample.group for sample in samples})
    metrics: list[GroupMetric] = []
    for group in groups:
        group_results = [
            results_by_id[sample.sample_id] for sample in samples if sample.group == group
        ]
        correct = sum(1 for result in group_results if result.correct)
        metrics.append(
            GroupMetric(
                group=group,
                accuracy=correct / len(group_results) if group_results else 0.0,
                count=len(group_results),
            )
        )
    return tuple(metrics)


def worst_group_accuracy(metrics: tuple[GroupMetric, ...]) -> float:
    """Return the lowest group accuracy."""

    if not metrics:
        return 0.0
    return min(metric.accuracy for metric in metrics)
