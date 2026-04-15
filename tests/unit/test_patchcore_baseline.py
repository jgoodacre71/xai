from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.models.patchcore import (
    TorchvisionBackbonePatchFeatureExtractor,
    build_mean_colour_memory_bank,
    build_patchcore_memory_bank,
    score_image_against_memory_bank,
    score_image_with_extractor,
    score_to_provenance_artefact,
)
from xai_demo_suite.models.patchcore.types import FloatArray


class ConstantPatchFeatureExtractor:
    feature_name = "constant_test"

    def extract(self, image_path: Path, boxes: list[BoundingBox]) -> FloatArray:
        del image_path
        features = np.empty((len(boxes), 2), dtype=np.float64)
        for index, box in enumerate(boxes):
            features[index] = (box.x, box.y)
        return features


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


def test_custom_extractor_uses_same_provenance_path(tmp_path: Path) -> None:
    nominal = tmp_path / "nominal.png"
    query = tmp_path / "query.png"
    _write_colour_image(nominal, (64, 64, 64), size=32)
    _write_colour_image(query, (128, 128, 128), size=32)
    extractor = ConstantPatchFeatureExtractor()

    memory_bank = build_patchcore_memory_bank(
        [_record(nominal)],
        extractor=extractor,
        patch_size=16,
        stride=16,
    )
    scores = score_image_with_extractor(
        sample_id="query",
        image_path=query,
        memory_bank=memory_bank,
        extractor=extractor,
        patch_size=16,
        stride=16,
        top_k=1,
    )

    assert memory_bank.feature_name == "constant_test"
    assert memory_bank.features.shape == (4, 2)
    assert scores[0].nearest[0].metadata.source_path == nominal
    assert scores[0].nearest[0].metadata.box == scores[0].query_box


def test_extractor_mismatch_is_rejected(tmp_path: Path) -> None:
    nominal = tmp_path / "nominal.png"
    query = tmp_path / "query.png"
    _write_colour_image(nominal, (64, 64, 64), size=32)
    _write_colour_image(query, (128, 128, 128), size=32)
    memory_bank = build_mean_colour_memory_bank([_record(nominal)], patch_size=32, stride=32)

    with pytest.raises(ValueError, match="feature name does not match"):
        score_image_with_extractor(
            sample_id="query",
            image_path=query,
            memory_bank=memory_bank,
            extractor=ConstantPatchFeatureExtractor(),
            patch_size=32,
            stride=32,
        )


def test_optional_torchvision_extractor_has_actionable_error_without_dependencies() -> None:
    if importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"):
        pytest.skip("Torch dependencies are installed in this environment.")

    with pytest.raises(RuntimeError, match="requires optional dependencies"):
        TorchvisionBackbonePatchFeatureExtractor()
