"""Simple deterministic classifiers for the synthetic shortcut lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from xai_demo_suite.data.synthetic import IndustrialShortcutSample
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """One classification result for a shortcut demo sample."""

    sample_id: str
    label: str
    predicted: str
    score: float
    correct: bool


class IndustrialClassifier(Protocol):
    """Protocol for deterministic industrial shortcut classifiers."""

    @property
    def name(self) -> str:
        """Stable classifier name used in reports."""
        ...

    def predict_score(self, image_path: Path) -> float:
        """Return a positive score for the defect class."""


def _region_array(image_path: Path, box: BoundingBox) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return rgb[box.y : box.y + box.height, box.x : box.x + box.width, :]


@dataclass(frozen=True, slots=True)
class StampShortcutClassifier:
    """Classifier that intentionally keys off the corner stamp colour."""

    name: str = "corner_stamp_shortcut"
    stamp_region: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=6, y=6, width=20, height=20)
    )

    def predict_score(self, image_path: Path) -> float:
        region = _region_array(image_path, self.stamp_region)
        red = float(region[:, :, 0].mean())
        blue = float(region[:, :, 2].mean())
        return red - blue


@dataclass(frozen=True, slots=True)
class ShapeClassifier:
    """Classifier that uses the central part silhouette instead of the stamp."""

    name: str = "central_shape_intervention"
    object_region: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=42, y=42, width=44, height=44)
    )

    def predict_score(self, image_path: Path) -> float:
        region = _region_array(image_path, self.object_region)
        occupied = np.linalg.norm(region - np.array([0.74, 0.80, 0.77]), axis=2) < 0.22
        row_counts = occupied.sum(axis=1)
        full_rows = int(np.count_nonzero(row_counts > 28))
        centre_column_count = int(np.count_nonzero(occupied[:, occupied.shape[1] // 2]))
        # Blocks have many full rows; discs have fewer full rows and a rounder outline.
        return float(full_rows - (centre_column_count * 0.15) - 20.0)


def predict_label(score: float) -> str:
    """Convert a defect score into a class label."""

    return "defect" if score > 0.0 else "normal"


def evaluate_classifier(
    classifier: IndustrialClassifier,
    samples: list[IndustrialShortcutSample],
) -> list[ClassificationResult]:
    """Evaluate a classifier on synthetic shortcut samples."""

    results: list[ClassificationResult] = []
    for sample in samples:
        score = classifier.predict_score(sample.image_path)
        predicted = predict_label(score)
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


def accuracy(results: list[ClassificationResult]) -> float:
    """Return classification accuracy."""

    if not results:
        return 0.0
    return sum(1 for result in results if result.correct) / len(results)


def mask_region(image_path: Path, box: BoundingBox, output_path: Path) -> Path:
    """Save a copy of an image with a region replaced by background colour."""

    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        panel = image.convert("RGB")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), fill=(44, 52, 58))
        panel.save(output_path)
    return output_path


def swap_stamp(image_path: Path, stamp: str, output_path: Path) -> Path:
    """Save a copy of an image with the corner stamp replaced."""

    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        panel = image.convert("RGB")
        draw = ImageDraw.Draw(panel)
        if stamp == "red":
            colour = (216, 70, 64)
        elif stamp == "blue":
            colour = (56, 116, 214)
        elif stamp == "none":
            colour = (44, 52, 58)
        else:
            raise ValueError(f"Unsupported stamp: {stamp}")
        draw.rectangle((6, 6, 26, 26), fill=colour)
        panel.save(output_path)
    return output_path
