"""Small PatchCore-style baseline focused on provenance.

This module is intentionally light. It defaults to mean RGB patch features, but
the scoring path accepts any patch feature extractor that follows the local
protocol. The purpose is to establish the memory-bank metadata, nearest-normal
retrieval, and explanation artefact plumbing before introducing a full PatchCore
implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.explain.contracts import BoundingBox, ProvenanceArtefact
from xai_demo_suite.models.patchcore.features import (
    MeanRGBPatchFeatureExtractor,
    PatchFeatureExtractor,
)
from xai_demo_suite.models.patchcore.types import (
    FloatArray,
    PatchCoreMemoryBank,
    PatchMetadata,
    PatchNearestNeighbour,
    PatchScore,
)


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return height, width


def _iter_patch_boxes(height: int, width: int, patch_size: int, stride: int) -> list[BoundingBox]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")
    if height < patch_size or width < patch_size:
        raise ValueError("Image is smaller than patch_size.")

    boxes: list[BoundingBox] = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            boxes.append(BoundingBox(x=x, y=y, width=patch_size, height=patch_size))
    return boxes


def build_patchcore_memory_bank(
    records: Sequence[ImageManifestRecord],
    *,
    extractor: PatchFeatureExtractor,
    patch_size: int = 32,
    stride: int = 32,
) -> PatchCoreMemoryBank:
    """Build a nominal memory bank from a patch feature extractor."""

    all_features: list[FloatArray] = []
    metadata: list[PatchMetadata] = []
    feature_vector_id = 0

    for record in records:
        if record.is_anomalous:
            raise ValueError("PatchCore memory banks must be built from nominal records only.")

        height, width = _image_dimensions(record.image_path)
        boxes = _iter_patch_boxes(height=height, width=width, patch_size=patch_size, stride=stride)
        features = extractor.extract(record.image_path, boxes)
        if features.shape[0] != len(boxes):
            raise ValueError("Extractor must return one feature row per patch box.")
        all_features.append(features)

        for patch_index, box in enumerate(boxes):
            metadata.append(
                PatchMetadata(
                    patch_id=f"{record.sample_id}/patch-{patch_index:05d}",
                    source_image_id=record.sample_id,
                    source_split=record.split,
                    source_path=record.image_path,
                    box=box,
                    feature_vector_id=feature_vector_id,
                )
            )
            feature_vector_id += 1

    if not all_features:
        raise ValueError("Cannot build a memory bank from no records.")

    return PatchCoreMemoryBank(
        features=np.vstack(all_features),
        metadata=tuple(metadata),
        feature_name=extractor.feature_name,
    )


def build_mean_colour_memory_bank(
    records: Sequence[ImageManifestRecord],
    *,
    patch_size: int = 32,
    stride: int = 32,
) -> PatchCoreMemoryBank:
    """Build a nominal memory bank from mean-colour patch features."""

    return build_patchcore_memory_bank(
        records,
        extractor=MeanRGBPatchFeatureExtractor(),
        patch_size=patch_size,
        stride=stride,
    )


def reduce_memory_bank_coreset(
    memory_bank: PatchCoreMemoryBank,
    *,
    max_patches: int,
    seed: int = 0,
) -> PatchCoreMemoryBank:
    """Reduce a memory bank with deterministic greedy k-centre selection."""

    if max_patches <= 0:
        raise ValueError("max_patches must be positive.")
    if memory_bank.features.shape[0] <= max_patches:
        return memory_bank

    features = memory_bank.features.astype(np.float64, copy=False)
    row_count = features.shape[0]
    rng = np.random.default_rng(seed)
    selected_indices = np.empty(max_patches, dtype=np.int64)
    selected_mask = np.zeros(row_count, dtype=bool)

    first_index = int(rng.integers(0, row_count))
    selected_indices[0] = first_index
    selected_mask[first_index] = True
    min_squared_distances = np.sum((features - features[first_index]) ** 2, axis=1)
    min_squared_distances[selected_mask] = -np.inf

    for output_index in range(1, max_patches):
        next_index = int(np.argmax(min_squared_distances))
        selected_indices[output_index] = next_index
        selected_mask[next_index] = True
        squared_distances = np.sum((features - features[next_index]) ** 2, axis=1)
        min_squared_distances = np.minimum(min_squared_distances, squared_distances)
        min_squared_distances[selected_mask] = -np.inf

    return PatchCoreMemoryBank(
        features=memory_bank.features[selected_indices],
        metadata=tuple(memory_bank.metadata[int(index)] for index in selected_indices),
        feature_name=memory_bank.feature_name,
    )


def _nearest_neighbours(
    query_feature: FloatArray,
    memory_bank: PatchCoreMemoryBank,
    top_k: int,
) -> tuple[PatchNearestNeighbour, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    distances = np.linalg.norm(memory_bank.features - query_feature, axis=1)
    nearest_indices = np.argsort(distances)[:top_k]
    return tuple(
        PatchNearestNeighbour(
            metadata=memory_bank.metadata[int(index)],
            distance=float(distances[int(index)]),
        )
        for index in nearest_indices
    )


def score_image_with_extractor(
    *,
    sample_id: str,
    image_path: Path,
    memory_bank: PatchCoreMemoryBank,
    extractor: PatchFeatureExtractor,
    patch_size: int = 32,
    stride: int = 32,
    top_k: int = 3,
) -> list[PatchScore]:
    """Score image patches by distance to the nominal memory bank."""

    if memory_bank.feature_name != extractor.feature_name:
        raise ValueError(
            "Memory-bank feature name does not match extractor feature name: "
            f"{memory_bank.feature_name!r} != {extractor.feature_name!r}"
        )

    height, width = _image_dimensions(image_path)
    boxes = _iter_patch_boxes(height=height, width=width, patch_size=patch_size, stride=stride)
    features = extractor.extract(image_path, boxes)
    if features.shape[0] != len(boxes):
        raise ValueError("Extractor must return one feature row per patch box.")

    scores: list[PatchScore] = []
    for box, feature in zip(boxes, features, strict=True):
        nearest = _nearest_neighbours(
            query_feature=feature,
            memory_bank=memory_bank,
            top_k=top_k,
        )
        scores.append(
            PatchScore(
                sample_id=sample_id,
                image_path=image_path,
                query_box=box,
                distance=nearest[0].distance,
                nearest=nearest,
            )
        )
    return sorted(scores, key=lambda score: score.distance, reverse=True)


def score_image_against_memory_bank(
    *,
    sample_id: str,
    image_path: Path,
    memory_bank: PatchCoreMemoryBank,
    patch_size: int = 32,
    stride: int = 32,
    top_k: int = 3,
) -> list[PatchScore]:
    """Score image patches using the default mean-colour baseline extractor."""

    return score_image_with_extractor(
        sample_id=sample_id,
        image_path=image_path,
        memory_bank=memory_bank,
        extractor=MeanRGBPatchFeatureExtractor(),
        patch_size=patch_size,
        stride=stride,
        top_k=top_k,
    )


def score_to_provenance_artefact(
    score: PatchScore,
    *,
    method: str = "mean-rgb-patchcore",
) -> ProvenanceArtefact:
    """Convert nearest-normal patch evidence into the shared provenance contract."""

    return ProvenanceArtefact(
        sample_id=score.sample_id,
        method=method,
        reference_ids=[neighbour.metadata.source_image_id for neighbour in score.nearest],
        reference_scores=[neighbour.distance for neighbour in score.nearest],
        reference_image_paths=[neighbour.metadata.source_path for neighbour in score.nearest],
        reference_boxes=[neighbour.metadata.box for neighbour in score.nearest],
        note=(
            "Nearest nominal patches from the current PatchCore-style memory bank. "
            "This baseline uses mean RGB patch features, not deep PatchCore features."
        ),
    )
