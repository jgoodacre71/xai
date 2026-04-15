"""Perturbation and explanation-drift helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class DriftMeasurement:
    """Prediction and explanation movement for one perturbation."""

    perturbation_name: str
    baseline_score: float
    perturbed_score: float
    baseline_prediction: str
    perturbed_prediction: str
    baseline_region: BoundingBox
    perturbed_region: BoundingBox

    @property
    def prediction_changed(self) -> bool:
        """Return whether the predicted class changed."""

        return self.baseline_prediction != self.perturbed_prediction

    @property
    def score_shift(self) -> float:
        """Return absolute score movement."""

        return abs(self.perturbed_score - self.baseline_score)

    @property
    def explanation_shift(self) -> float:
        """Return centre-point movement between explanation regions."""

        baseline_x = self.baseline_region.x + (self.baseline_region.width / 2.0)
        baseline_y = self.baseline_region.y + (self.baseline_region.height / 2.0)
        perturbed_x = self.perturbed_region.x + (self.perturbed_region.width / 2.0)
        perturbed_y = self.perturbed_region.y + (self.perturbed_region.height / 2.0)
        return hypot(perturbed_x - baseline_x, perturbed_y - baseline_y)


def perturb_image(image_path: Path, output_path: Path, perturbation_name: str) -> Path:
    """Write a deterministic perturbed image."""

    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        panel = image.convert("RGB")
        if perturbation_name == "brightness_up":
            panel = ImageEnhance.Brightness(panel).enhance(1.35)
        elif perturbation_name == "contrast_down":
            panel = ImageEnhance.Contrast(panel).enhance(0.65)
        elif perturbation_name == "blur":
            panel = panel.filter(ImageFilter.GaussianBlur(radius=1.4))
        else:
            raise ValueError(f"Unsupported perturbation: {perturbation_name}")
        panel.save(output_path)
    return output_path
