"""Patch feature extraction interfaces for PatchCore-style models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.models.patchcore.types import FloatArray


class PatchFeatureExtractor(Protocol):
    """Protocol for patch-level feature extractors."""

    @property
    def feature_name(self) -> str:
        """Short stable name stored with the memory bank."""

    def extract(self, image_path: Path, boxes: list[BoundingBox]) -> FloatArray:
        """Return one feature row per patch box."""


def load_rgb_array(image_path: Path) -> FloatArray:
    """Load an image as a float RGB array in the range [0, 1]."""

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float64) / 255.0


@dataclass(frozen=True, slots=True)
class MeanRGBPatchFeatureExtractor:
    """Mean RGB patch features used as a deterministic baseline."""

    feature_name: str = "mean_rgb"

    def extract(self, image_path: Path, boxes: list[BoundingBox]) -> FloatArray:
        """Extract mean RGB features for each requested image patch."""

        image = load_rgb_array(image_path)
        features = np.empty((len(boxes), 3), dtype=np.float64)
        for index, box in enumerate(boxes):
            patch = image[box.y : box.y + box.height, box.x : box.x + box.width, :]
            features[index] = patch.mean(axis=(0, 1))
        return features


@dataclass(frozen=True, slots=True)
class TorchvisionBackbonePatchFeatureExtractor:
    """Optional Torch/Torchvision extractor for future deep PatchCore work.

    The class imports Torch lazily inside ``__post_init__`` so the base package
    and tests do not require heavyweight ML dependencies. It currently provides
    a clear integration point; using pretrained weights and multi-scale feature
    maps should be added in a later task.
    """

    backbone_name: str = "resnet18"
    feature_name: str = "torchvision_resnet18"

    def __post_init__(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
            import torchvision  # type: ignore[import-not-found]  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TorchvisionBackbonePatchFeatureExtractor requires optional "
                "dependencies 'torch' and 'torchvision'. Install them before "
                "using the deep feature path."
            ) from exc

    def extract(self, image_path: Path, boxes: list[BoundingBox]) -> FloatArray:
        """Extract deep patch features.

        This placeholder intentionally fails until the Torch feature task adds a
        concrete backbone pipeline. The class exists now to make dependency and
        interface boundaries explicit.
        """

        raise NotImplementedError(
            f"Deep feature extraction for {self.backbone_name} is not implemented yet."
        )
