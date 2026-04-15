"""PatchCore-style anomaly detection helpers."""

from xai_demo_suite.models.patchcore.baseline import (
    build_mean_colour_memory_bank,
    score_image_against_memory_bank,
    score_to_provenance_artefact,
)
from xai_demo_suite.models.patchcore.types import (
    PatchCoreMemoryBank,
    PatchMetadata,
    PatchNearestNeighbour,
    PatchScore,
)

__all__ = [
    "PatchCoreMemoryBank",
    "PatchMetadata",
    "PatchNearestNeighbour",
    "PatchScore",
    "build_mean_colour_memory_bank",
    "score_image_against_memory_bank",
    "score_to_provenance_artefact",
]
