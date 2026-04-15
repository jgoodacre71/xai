from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.models.patchcore.types import FloatArray
from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    build_patchcore_bottle_report,
)
from xai_demo_suite.vis.image_panels import draw_box_on_image, save_patch_crop


class ConstantPatchFeatureExtractor:
    feature_name = "constant_report_test"

    def extract(self, image_path: Path, boxes: list[BoundingBox]) -> FloatArray:
        del image_path
        output = np.empty((len(boxes), 2), dtype=np.float64)
        for index, box in enumerate(boxes):
            output[index] = (box.x, box.y)
        return output


def _write_image(path: Path, colour: tuple[int, int, int], size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, :] = colour
    image[8:16, 8:16, :] = (255, 0, 0)
    Image.fromarray(image, mode="RGB").save(path)


def _write_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "data" / "processed" / "mvtec_ad" / "bottle" / "manifest.jsonl"
    train_path = (
        tmp_path
        / "data"
        / "interim"
        / "mvtec_ad"
        / "bottle"
        / "train"
        / "good"
        / "000.png"
    )
    test_path = (
        tmp_path / "data" / "interim" / "mvtec_ad" / "bottle" / "test" / "broken" / "001.png"
    )
    _write_image(train_path, (30, 30, 30))
    _write_image(test_path, (200, 200, 200))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": "mvtec_ad",
            "category": "bottle",
            "split": "train",
            "defect_type": "good",
            "is_anomalous": False,
            "image_path": "data/interim/mvtec_ad/bottle/train/good/000.png",
            "mask_path": None,
        },
        {
            "dataset": "mvtec_ad",
            "category": "bottle",
            "split": "test",
            "defect_type": "broken",
            "is_anomalous": True,
            "image_path": "data/interim/mvtec_ad/bottle/test/broken/001.png",
            "mask_path": None,
        },
    ]
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def test_save_patch_crop_uses_recorded_coordinates(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_image(image_path, (0, 0, 0), size=32)
    crop_path = save_patch_crop(
        image_path=image_path,
        box=BoundingBox(x=8, y=8, width=8, height=8),
        output_path=tmp_path / "crop.png",
        scale=1,
    )

    with Image.open(crop_path) as crop:
        assert crop.size == (8, 8)
        assert crop.getpixel((0, 0)) == (255, 0, 0)


def test_draw_box_on_image_writes_panel(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_image(image_path, (0, 0, 0), size=32)
    panel_path = draw_box_on_image(
        image_path=image_path,
        box=BoundingBox(x=4, y=4, width=8, height=8),
        output_path=tmp_path / "panel.png",
        colour=(1, 2, 3),
        width=1,
    )

    with Image.open(panel_path) as panel:
        assert panel.getpixel((4, 4)) == (1, 2, 3)


def test_patchcore_bottle_report_writes_html_and_assets(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    config = PatchCoreBottleReportConfig(
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        cache_path=tmp_path / "artefacts" / "bank.npz",
        patch_size=16,
        stride=16,
        top_k=1,
        use_cache=False,
    )

    output_path = build_patchcore_bottle_report(
        config,
        extractor=ConstantPatchFeatureExtractor(),
    )

    html = output_path.read_text(encoding="utf-8")
    assert "PatchCore Bottle Report" in html
    assert "Nearest Normal Patch Evidence" in html
    assert (config.output_dir / "assets" / "query_patch.png").exists()
    assert (config.output_dir / "assets" / "normal_patch_1.png").exists()
