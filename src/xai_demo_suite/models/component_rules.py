"""Component-aware template comparators for fixed-layout packaging demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.explain.contracts import BoundingBox


@dataclass(frozen=True, slots=True)
class ComponentTemplate:
    """A learned nominal template for one expected component region."""

    name: str
    box: BoundingBox
    template: NDArray[np.float32]
    display_template: NDArray[np.float32]
    threshold: float
    train_mean_score: float
    train_std_score: float


@dataclass(frozen=True, slots=True)
class ComponentRuleScore:
    """A component-template score for one image."""

    name: str
    box: BoundingBox
    score: float
    z_score: float
    threshold: float
    flagged: bool


def _crop_array(image_path: Path, box: BoundingBox) -> NDArray[np.float32]:
    with Image.open(image_path) as image:
        crop = image.convert("RGB").crop((box.x, box.y, box.x + box.width, box.y + box.height))
    return np.array(crop, dtype=np.float32)


def _normalise_region(region: NDArray[np.float32]) -> NDArray[np.float32]:
    grayscale = region.mean(axis=2) / 255.0
    grayscale = grayscale - float(grayscale.mean())
    scale = float(grayscale.std())
    if scale < 1e-6:
        scale = 1.0
    return np.asarray(grayscale / scale, dtype=np.float32)


def juice_bottle_front_label_box(image_size: tuple[int, int]) -> BoundingBox:
    """Return the expected front-label region for aligned juice-bottle images."""

    width, height = image_size
    return BoundingBox(
        x=round(width * 0.31),
        y=round(height * 0.43),
        width=round(width * 0.38),
        height=round(height * 0.23),
    )


def fit_juice_bottle_front_label_template(
    train_records: list[ImageManifestRecord],
) -> ComponentTemplate:
    """Fit a simple template comparator over the aligned front-label region."""

    if not train_records:
        raise ValueError("At least one nominal train record is required.")

    with Image.open(train_records[0].image_path) as image:
        box = juice_bottle_front_label_box(image.size)

    raw_regions = [_crop_array(record.image_path, box) for record in train_records]
    normalised_regions = [_normalise_region(region) for region in raw_regions]
    template = np.mean(normalised_regions, axis=0)
    display_template = np.mean(raw_regions, axis=0)
    train_scores = np.array(
        [float(np.mean(np.abs(region - template))) for region in normalised_regions],
        dtype=np.float64,
    )
    train_mean = float(train_scores.mean())
    train_std = float(train_scores.std())
    threshold = max(float(np.quantile(train_scores, 0.95)), train_mean + 3.0 * train_std)
    return ComponentTemplate(
        name="front_label",
        box=box,
        template=template,
        display_template=display_template,
        threshold=threshold,
        train_mean_score=train_mean,
        train_std_score=train_std,
    )


def score_component_template(
    template: ComponentTemplate,
    record: ImageManifestRecord,
) -> ComponentRuleScore:
    """Score one image against a learned component template."""

    region = _crop_array(record.image_path, template.box)
    normalised = _normalise_region(region)
    score = float(np.mean(np.abs(normalised - template.template)))
    scale = template.train_std_score if template.train_std_score > 1e-6 else 1.0
    z_score = (score - template.train_mean_score) / scale
    return ComponentRuleScore(
        name=template.name,
        box=template.box,
        score=score,
        z_score=float(z_score),
        threshold=template.threshold,
        flagged=score > template.threshold,
    )
