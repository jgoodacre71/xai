"""Memory-bank cache helpers for PatchCore artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchMetadata
from xai_demo_suite.utils.io import ensure_directory


def save_memory_bank(memory_bank: PatchCoreMemoryBank, cache_path: Path) -> Path:
    """Save memory-bank features and provenance metadata to ``cache_path``."""

    ensure_directory(cache_path.parent)
    metadata = [
        {
            "patch_id": item.patch_id,
            "source_image_id": item.source_image_id,
            "source_split": item.source_split,
            "source_path": item.source_path.as_posix(),
            "box": {
                "x": item.box.x,
                "y": item.box.y,
                "width": item.box.width,
                "height": item.box.height,
            },
            "feature_vector_id": item.feature_vector_id,
        }
        for item in memory_bank.metadata
    ]
    np.savez_compressed(
        cache_path,
        features=memory_bank.features,
        feature_name=np.array(memory_bank.feature_name),
        metadata_json=np.array(json.dumps(metadata)),
    )
    return cache_path


def load_memory_bank(cache_path: Path) -> PatchCoreMemoryBank:
    """Load a memory bank written by :func:`save_memory_bank`."""

    with np.load(cache_path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float64)
        feature_name = str(archive["feature_name"].item())
        metadata_raw = str(archive["metadata_json"].item())

    metadata = []
    for item in json.loads(metadata_raw):
        box = item["box"]
        metadata.append(
            PatchMetadata(
                patch_id=str(item["patch_id"]),
                source_image_id=str(item["source_image_id"]),
                source_split=str(item["source_split"]),
                source_path=Path(str(item["source_path"])),
                box=BoundingBox(
                    x=int(box["x"]),
                    y=int(box["y"]),
                    width=int(box["width"]),
                    height=int(box["height"]),
                ),
                feature_vector_id=int(item["feature_vector_id"]),
            )
        )

    return PatchCoreMemoryBank(
        features=features,
        metadata=tuple(metadata),
        feature_name=feature_name,
    )
