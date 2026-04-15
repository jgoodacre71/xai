"""Image panel helpers used by reports and notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.models.patchcore.types import PatchScore
from xai_demo_suite.utils.io import ensure_directory


def _crop_box(box: BoundingBox) -> tuple[int, int, int, int]:
    return (box.x, box.y, box.x + box.width, box.y + box.height)


def save_patch_crop(
    *,
    image_path: Path,
    box: BoundingBox,
    output_path: Path,
    scale: int = 4,
) -> Path:
    """Save a crop from ``image_path`` using the recorded source coordinates."""

    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        crop = image.convert("RGB").crop(_crop_box(box))
        if scale > 1:
            crop = crop.resize(
                (box.width * scale, box.height * scale),
                Image.Resampling.NEAREST,
            )
        crop.save(output_path)
    return output_path


def draw_box_on_image(
    *,
    image_path: Path,
    box: BoundingBox,
    output_path: Path,
    colour: tuple[int, int, int] = (220, 20, 60),
    width: int = 4,
) -> Path:
    """Save an image copy with a bounding box overlay."""

    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        panel = image.convert("RGB")
        draw = ImageDraw.Draw(panel)
        for offset in range(width):
            draw.rectangle(
                (
                    box.x - offset,
                    box.y - offset,
                    box.x + box.width + offset,
                    box.y + box.height + offset,
                ),
                outline=colour,
            )
        panel.save(output_path)
    return output_path


def normalise_patch_scores(scores: list[PatchScore]) -> list[float]:
    """Normalise patch distances into the range [0, 1]."""

    if not scores:
        return []
    distances = np.array([score.distance for score in scores], dtype=np.float64)
    minimum = float(distances.min())
    maximum = float(distances.max())
    if maximum == minimum:
        return [0.0 for _ in scores]
    normalised = (distances - minimum) / (maximum - minimum)
    return [float(value) for value in normalised]


def _score_colour(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    return (
        int(255 * value),
        int(40 * (1.0 - value)),
        int(255 * (1.0 - value)),
    )


def save_score_overlay(
    *,
    image_path: Path,
    scores: list[PatchScore],
    output_path: Path,
    alpha: float = 0.45,
) -> Path:
    """Save a coarse patch-score overlay for an image."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [0, 1].")

    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        base = image.convert("RGB")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for score, value in zip(scores, normalise_patch_scores(scores), strict=True):
            colour = _score_colour(value)
            draw.rectangle(
                _crop_box(score.query_box),
                fill=(*colour, int(255 * alpha)),
            )
        blended = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        blended.save(output_path)
    return output_path
