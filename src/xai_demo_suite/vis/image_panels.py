"""Image panel helpers used by reports and notebooks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from xai_demo_suite.explain.contracts import BoundingBox
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
