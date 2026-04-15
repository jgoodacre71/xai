from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.models.patchcore import (
    build_mean_colour_memory_bank,
    score_image_against_memory_bank,
    score_to_provenance_artefact,
)


def _write_colour_image(path: Path, colour: tuple[int, int, int], size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, :] = colour
    Image.fromarray(image, mode="RGB").save(path)


def _record(path: Path, *, split: str = "train", defect_type: str = "good") -> ImageManifestRecord:
    return ImageManifestRecord(
        dataset="mvtec_ad",
        category="bottle",
        split=split,
        defect_type=defect_type,
        is_anomalous=defect_type != "good",
        image_path=path,
        mask_path=None,
    )


def test_memory_bank_preserves_patch_source_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "nominal.png"
    _write_colour_image(image_path, (10, 20, 30), size=32)

    memory_bank = build_mean_colour_memory_bank(
        [_record(image_path)],
        patch_size=16,
        stride=16,
    )

    assert memory_bank.features.shape == (4, 3)
    assert len(memory_bank.metadata) == 4
    assert memory_bank.metadata[0].source_image_id == "mvtec_ad/bottle/train/good/nominal"
    assert memory_bank.metadata[0].source_path == image_path
    assert memory_bank.metadata[0].box.width == 16
    assert memory_bank.metadata[0].feature_vector_id == 0


def test_score_image_returns_nearest_normal_patch_evidence(tmp_path: Path) -> None:
    nominal_dark = tmp_path / "nominal_dark.png"
    nominal_bright = tmp_path / "nominal_bright.png"
    query = tmp_path / "query.png"
    _write_colour_image(nominal_dark, (0, 0, 0), size=32)
    _write_colour_image(nominal_bright, (255, 255, 255), size=32)
    _write_colour_image(query, (250, 250, 250), size=32)

    memory_bank = build_mean_colour_memory_bank(
        [_record(nominal_dark), _record(nominal_bright)],
        patch_size=32,
        stride=32,
    )
    scores = score_image_against_memory_bank(
        sample_id="query",
        image_path=query,
        memory_bank=memory_bank,
        patch_size=32,
        stride=32,
        top_k=2,
    )

    assert len(scores) == 1
    assert scores[0].nearest[0].metadata.source_path == nominal_bright
    assert scores[0].nearest[0].distance < scores[0].nearest[1].distance


def test_score_to_provenance_artefact_keeps_source_boxes(tmp_path: Path) -> None:
    nominal = tmp_path / "nominal.png"
    query = tmp_path / "query.png"
    _write_colour_image(nominal, (64, 64, 64), size=32)
    _write_colour_image(query, (80, 80, 80), size=32)

    memory_bank = build_mean_colour_memory_bank([_record(nominal)], patch_size=32, stride=32)
    score = score_image_against_memory_bank(
        sample_id="query",
        image_path=query,
        memory_bank=memory_bank,
        patch_size=32,
        stride=32,
        top_k=1,
    )[0]

    artefact = score_to_provenance_artefact(score)

    assert artefact.sample_id == "query"
    assert artefact.reference_ids == ["mvtec_ad/bottle/train/good/nominal"]
    assert artefact.reference_image_paths == [nominal]
    assert artefact.reference_boxes is not None
    assert artefact.reference_boxes[0].area == 1024
