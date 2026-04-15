"""Small PatchCore-style baseline focused on provenance.

This module is intentionally light: it uses mean RGB patch features rather than
deep backbone features. The purpose is to establish the memory-bank metadata,
nearest-normal retrieval, and explanation artefact plumbing before introducing a
full PatchCore implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.explain.contracts import BoundingBox, ProvenanceArtefact
from xai_demo_suite.models.patchcore.types import (
    FloatArray,
    PatchCoreMemoryBank,
    PatchMetadata,
    PatchNearestNeighbour,
    PatchScore,
)


def _load_rgb_array(image_path: Path) -> FloatArray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float64) / 255.0


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


def _extract_mean_colour_features(
    image: FloatArray,
    boxes: Sequence[BoundingBox],
) -> FloatArray:
    features = np.empty((len(boxes), 3), dtype=np.float64)
    for index, box in enumerate(boxes):
        patch = image[box.y : box.y + box.height, box.x : box.x + box.width, :]
        features[index] = patch.mean(axis=(0, 1))
    return features


def build_mean_colour_memory_bank(
    records: Sequence[ImageManifestRecord],
    *,
    patch_size: int = 32,
    stride: int = 32,
) -> PatchCoreMemoryBank:
    """Build a nominal memory bank from mean-colour patch features."""

    all_features: list[FloatArray] = []
    metadata: list[PatchMetadata] = []
    feature_vector_id = 0

    for record in records:
        if record.is_anomalous:
            raise ValueError("PatchCore memory banks must be built from nominal records only.")

        image = _load_rgb_array(record.image_path)
        height, width = image.shape[:2]
        boxes = _iter_patch_boxes(height=height, width=width, patch_size=patch_size, stride=stride)
        features = _extract_mean_colour_features(image=image, boxes=boxes)
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
        feature_name="mean_rgb",
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


def score_image_against_memory_bank(
    *,
    sample_id: str,
    image_path: Path,
    memory_bank: PatchCoreMemoryBank,
    patch_size: int = 32,
    stride: int = 32,
    top_k: int = 3,
) -> list[PatchScore]:
    """Score image patches by distance to the nominal memory bank."""

    image = _load_rgb_array(image_path)
    height, width = image.shape[:2]
    boxes = _iter_patch_boxes(height=height, width=width, patch_size=patch_size, stride=stride)
    features = _extract_mean_colour_features(image=image, boxes=boxes)

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
