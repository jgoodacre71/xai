"""Typed PatchCore provenance and scoring records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from xai_demo_suite.explain.contracts import BoundingBox

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PatchMetadata:
    """Source metadata for one retained nominal patch."""

    patch_id: str
    source_image_id: str
    source_split: str
    source_path: Path
    box: BoundingBox
    feature_vector_id: int


@dataclass(frozen=True, slots=True)
class PatchCoreMemoryBank:
    """Patch feature matrix plus source metadata needed for provenance."""

    features: FloatArray
    metadata: tuple[PatchMetadata, ...]
    feature_name: str

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("Memory-bank features must be a 2D array.")
        if self.features.shape[0] != len(self.metadata):
            raise ValueError("Feature rows must match metadata records.")


@dataclass(frozen=True, slots=True)
class PatchNearestNeighbour:
    """Nearest nominal patch evidence for a scored query patch."""

    metadata: PatchMetadata
    distance: float


@dataclass(frozen=True, slots=True)
class PatchScore:
    """Scoring output for one query patch."""

    sample_id: str
    image_path: Path
    query_box: BoundingBox
    distance: float
    nearest: tuple[PatchNearestNeighbour, ...]
