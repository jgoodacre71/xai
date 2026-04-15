"""Patch feature extraction interfaces for PatchCore-style models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
class ColourTexturePatchFeatureExtractor:
    """Deterministic colour and texture patch features for local demos."""

    feature_name: str = "colour_texture"

    def extract(self, image_path: Path, boxes: list[BoundingBox]) -> FloatArray:
        """Extract low-dimensional colour, intensity, and edge statistics."""

        image = load_rgb_array(image_path)
        grayscale = image @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
        features = np.empty((len(boxes), 28), dtype=np.float64)
        for index, box in enumerate(boxes):
            patch_rgb = image[box.y : box.y + box.height, box.x : box.x + box.width, :]
            patch_gray = grayscale[box.y : box.y + box.height, box.x : box.x + box.width]
            features[index] = self._extract_patch_features(patch_rgb, patch_gray)
        return features

    def _extract_patch_features(self, patch_rgb: FloatArray, patch_gray: FloatArray) -> FloatArray:
        gradient_y, gradient_x = np.gradient(patch_gray)
        gradient = np.sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y))
        intensity_histogram, _ = np.histogram(
            patch_gray,
            bins=8,
            range=(0.0, 1.0),
            density=False,
        )
        gradient_histogram, _ = np.histogram(
            np.clip(gradient, 0.0, 0.25),
            bins=5,
            range=(0.0, 0.25),
            density=False,
        )
        patch_area = float(patch_gray.size)
        grey_quantiles = np.quantile(patch_gray, [0.10, 0.25, 0.50, 0.75, 0.90])
        return np.concatenate(
            [
                patch_rgb.mean(axis=(0, 1)) * 2.0,
                patch_rgb.std(axis=(0, 1)) * 4.0,
                np.array(
                    [
                        patch_gray.mean(),
                        patch_gray.std(),
                        gradient.mean(),
                        gradient.std(),
                    ],
                    dtype=np.float64,
                )
                * 4.0,
                grey_quantiles * 2.0,
                intensity_histogram.astype(np.float64) / patch_area * 3.0,
                gradient_histogram.astype(np.float64) / patch_area * 3.0,
            ]
        )


@dataclass(frozen=True, slots=True)
class TorchvisionBackbonePatchFeatureExtractor:
    """Torch/Torchvision ResNet patch-crop feature extractor.

    The class imports Torch lazily inside ``__post_init__`` so the base package
    and tests do not require heavyweight ML dependencies. It defaults to random
    weights to avoid implicit downloads. Pass ``weights_name="DEFAULT"`` only
    when an explicit pretrained-weight download/use policy is acceptable.
    """

    backbone_name: str = "resnet18"
    feature_name: str = "torchvision_resnet18"
    input_size: int = 224
    batch_size: int = 16
    weights_name: str | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
            import torchvision  # type: ignore[import-untyped]  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TorchvisionBackbonePatchFeatureExtractor requires optional "
                "dependencies 'torch' and 'torchvision'. Install them before "
                "using the deep feature path."
            ) from exc
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def extract(self, image_path: Path, boxes: list[BoundingBox]) -> FloatArray:
        """Extract one deep feature vector per requested patch box."""

        if not boxes:
            return np.empty((0, 0), dtype=np.float64)

        import torch
        from torchvision import models

        model = self._build_backbone(models=models, torch=torch)
        model.eval()
        model.to(self.device)

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            batches = [
                boxes[index : index + self.batch_size]
                for index in range(0, len(boxes), self.batch_size)
            ]
            output_features: list[FloatArray] = []
            with torch.no_grad():
                for batch_boxes in batches:
                    batch_tensor = torch.stack(
                        [
                            self._patch_to_tensor(
                                image=rgb_image,
                                box=box,
                                torch=torch,
                            )
                            for box in batch_boxes
                        ]
                    ).to(self.device)
                    batch_features = model(batch_tensor).flatten(start_dim=1)
                    output_features.append(
                        batch_features.detach().cpu().numpy().astype(np.float64)
                    )

        return np.vstack(output_features)

    def _build_backbone(self, *, models: Any, torch: Any) -> Any:
        if self.backbone_name != "resnet18":
            raise ValueError("Only resnet18 is currently supported.")

        weights = None
        if self.weights_name is not None:
            weights_enum = models.ResNet18_Weights
            weights = getattr(weights_enum, self.weights_name)

        resnet = models.resnet18(weights=weights)
        return torch.nn.Sequential(*list(resnet.children())[:-1])

    def _patch_to_tensor(self, *, image: Image.Image, box: BoundingBox, torch: Any) -> Any:
        crop = image.crop((box.x, box.y, box.x + box.width, box.y + box.height))
        crop = crop.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        array = np.asarray(crop, dtype=np.float32)
        array = np.divide(array, np.float32(255.0))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        array = np.divide(np.subtract(array, mean), std)
        return torch.from_numpy(array.transpose(2, 0, 1))
