"""Perturbation and explanation-drift helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

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
        elif perturbation_name == "lighting_warm":
            red, green, blue = panel.split()
            red = red.point(lambda value: min(255, int(value * 1.12)))
            blue = blue.point(lambda value: int(value * 0.88))
            panel = Image.merge("RGB", (red, green, blue))
        elif perturbation_name == "contrast_down":
            panel = ImageEnhance.Contrast(panel).enhance(0.65)
        elif perturbation_name == "blur":
            panel = panel.filter(ImageFilter.GaussianBlur(radius=1.4))
        elif perturbation_name == "jpeg_low_quality":
            panel.save(output_path, format="JPEG", quality=18, optimize=False)
            return output_path
        elif perturbation_name == "shadow_band":
            shadow = Image.new("RGBA", panel.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(shadow)
            draw.rectangle((0, 0, panel.width // 3, panel.height), fill=(0, 0, 0, 92))
            panel = Image.alpha_composite(panel.convert("RGBA"), shadow).convert("RGB")
        else:
            raise ValueError(f"Unsupported perturbation: {perturbation_name}")
        panel.save(output_path)
    return output_path


def normalise_heatmap(values: np.ndarray) -> np.ndarray:
    """Return a heatmap normalised into [0, 1]."""

    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def heatmap_drift(baseline: np.ndarray, perturbed: np.ndarray) -> float:
    """Return mean absolute difference between two normalised heatmaps."""

    baseline_normalised = normalise_heatmap(baseline)
    perturbed_normalised = normalise_heatmap(perturbed)
    return float(np.abs(baseline_normalised - perturbed_normalised).mean())
