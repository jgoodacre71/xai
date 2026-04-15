"""PatchCore-style anomaly detection helpers."""

from xai_demo_suite.models.patchcore.baseline import (
    build_mean_colour_memory_bank,
    build_patchcore_memory_bank,
    score_image_against_memory_bank,
    score_image_with_extractor,
    score_to_provenance_artefact,
)
from xai_demo_suite.models.patchcore.cache import load_memory_bank, save_memory_bank
from xai_demo_suite.models.patchcore.features import (
    MeanRGBPatchFeatureExtractor,
    PatchFeatureExtractor,
    TorchvisionBackbonePatchFeatureExtractor,
)
from xai_demo_suite.models.patchcore.types import (
    PatchCoreMemoryBank,
    PatchMetadata,
    PatchNearestNeighbour,
    PatchScore,
)

__all__ = [
    "MeanRGBPatchFeatureExtractor",
    "PatchCoreMemoryBank",
    "PatchFeatureExtractor",
    "PatchMetadata",
    "PatchNearestNeighbour",
    "PatchScore",
    "TorchvisionBackbonePatchFeatureExtractor",
    "build_mean_colour_memory_bank",
    "build_patchcore_memory_bank",
    "load_memory_bank",
    "save_memory_bank",
    "score_image_against_memory_bank",
    "score_image_with_extractor",
    "score_to_provenance_artefact",
]
