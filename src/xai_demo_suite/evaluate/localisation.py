"""Localisation checks for patch explanations against binary masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from xai_demo_suite.explain.contracts import BoundingBox

BoolMask = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PatchMaskOverlap:
    """Overlap between one scored patch and a binary ground-truth mask."""

    mask_path: Path
    patch_box: BoundingBox
    mask_area: int
    patch_area: int
    intersection_area: int

    @property
    def intersects_mask(self) -> bool:
        """Return whether the patch overlaps any positive mask pixel."""

        return self.intersection_area > 0

    @property
    def patch_mask_fraction(self) -> float:
        """Return the fraction of the patch covered by the ground-truth mask."""

        return self.intersection_area / self.patch_area if self.patch_area else 0.0

    @property
    def mask_covered_fraction(self) -> float:
        """Return the fraction of the full mask covered by the patch."""

        return self.intersection_area / self.mask_area if self.mask_area else 0.0


def load_binary_mask(mask_path: Path, target_size: tuple[int, int] | None = None) -> BoolMask:
    """Load a ground-truth mask as a boolean array.

    ``target_size`` follows PIL's ``(width, height)`` convention. If supplied,
    the mask is resized with nearest-neighbour interpolation before thresholding.
    """

    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        if target_size is not None and mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 0


def _target_size_from_image(image_path: Path | None) -> tuple[int, int] | None:
    if image_path is None:
        return None
    with Image.open(image_path) as image:
        return image.size


def measure_patch_mask_overlap(
    *,
    mask_path: Path,
    patch_box: BoundingBox,
    image_path: Path | None = None,
) -> PatchMaskOverlap:
    """Measure how a patch overlaps a binary ground-truth mask."""

    mask = load_binary_mask(
        mask_path=mask_path,
        target_size=_target_size_from_image(image_path),
    )
    height, width = mask.shape
    if patch_box.x + patch_box.width > width or patch_box.y + patch_box.height > height:
        raise ValueError("patch_box extends beyond the mask dimensions.")

    patch_mask = mask[
        patch_box.y : patch_box.y + patch_box.height,
        patch_box.x : patch_box.x + patch_box.width,
    ]
    intersection_area = int(np.count_nonzero(patch_mask))
    mask_area = int(np.count_nonzero(mask))
    return PatchMaskOverlap(
        mask_path=mask_path,
        patch_box=patch_box,
        mask_area=mask_area,
        patch_area=patch_box.area,
        intersection_area=intersection_area,
    )
