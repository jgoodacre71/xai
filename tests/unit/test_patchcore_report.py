from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from xai_demo_suite.evaluate.localisation import load_binary_mask, measure_patch_mask_overlap
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.explain.counterfactuals import (
    make_patch_replacement_artefact,
    replace_patch_from_source,
)
from xai_demo_suite.models.patchcore.types import (
    FloatArray,
    PatchMetadata,
    PatchNearestNeighbour,
    PatchScore,
)
from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    _roc_auc,
    build_patchcore_bottle_report,
)
from xai_demo_suite.vis.image_panels import (
    draw_box_on_image,
    normalise_patch_scores,
    save_mask_overlay,
    save_patch_crop,
    save_score_overlay,
)


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


def _write_mask(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    Image.fromarray(mask, mode="L").save(path)


def _write_manifest(
    tmp_path: Path,
    anomalous_count: int = 1,
    include_masks: bool = False,
    category: str = "bottle",
) -> Path:
    manifest_path = tmp_path / "data" / "processed" / "mvtec_ad" / category / "manifest.jsonl"
    train_path = (
        tmp_path
        / "data"
        / "interim"
        / "mvtec_ad"
        / category
        / "train"
        / "good"
        / "000.png"
    )
    _write_image(train_path, (30, 30, 30))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": "mvtec_ad",
            "category": category,
            "split": "train",
            "defect_type": "good",
            "is_anomalous": False,
            "image_path": f"data/interim/mvtec_ad/{category}/train/good/000.png",
            "mask_path": None,
        }
    ]
    good_test_path = (
        tmp_path
        / "data"
        / "interim"
        / "mvtec_ad"
        / category
        / "test"
        / "good"
        / "000.png"
    )
    _write_image(good_test_path, (30, 30, 30))
    rows.append(
        {
            "dataset": "mvtec_ad",
            "category": category,
            "split": "test",
            "defect_type": "good",
            "is_anomalous": False,
            "image_path": f"data/interim/mvtec_ad/{category}/test/good/000.png",
            "mask_path": None,
        }
    )
    for index in range(1, anomalous_count + 1):
        test_path = (
            tmp_path
            / "data"
            / "interim"
            / "mvtec_ad"
            / category
            / "test"
            / "broken"
            / f"{index:03d}.png"
        )
        _write_image(test_path, (200, 200, 200))
        mask_path = (
            tmp_path
            / "data"
            / "interim"
            / "mvtec_ad"
            / category
            / "ground_truth"
            / "broken"
            / f"{index:03d}_mask.png"
        )
        if include_masks:
            _write_mask(mask_path)
        rows.append(
            {
                "dataset": "mvtec_ad",
                "category": category,
                "split": "test",
                "defect_type": "broken",
                "is_anomalous": True,
                "image_path": (
                    f"data/interim/mvtec_ad/{category}/test/broken/{index:03d}.png"
                ),
                "mask_path": (
                    f"data/interim/mvtec_ad/{category}/ground_truth/broken/{index:03d}_mask.png"
                    if include_masks
                    else None
                ),
            }
        )
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


def test_load_binary_mask_and_patch_overlap(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, size=32)

    mask = load_binary_mask(mask_path)
    overlap = measure_patch_mask_overlap(
        mask_path=mask_path,
        patch_box=BoundingBox(x=0, y=0, width=16, height=16),
    )

    assert int(np.count_nonzero(mask)) == 64
    assert overlap.intersects_mask is True
    assert overlap.intersection_area == 64
    assert overlap.patch_area == 256
    assert overlap.mask_area == 64
    assert overlap.patch_mask_fraction == 0.25
    assert overlap.mask_covered_fraction == 1.0


def test_save_mask_overlay_writes_ground_truth_panel(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    _write_image(image_path, (10, 10, 10), size=32)
    _write_mask(mask_path, size=32)

    overlay_path = save_mask_overlay(
        image_path=image_path,
        mask_path=mask_path,
        output_path=tmp_path / "mask_overlay.png",
    )

    with Image.open(overlay_path) as overlay:
        assert overlay.size == (32, 32)
        assert overlay.getpixel((10, 10)) != (10, 10, 10)


def _patch_score(distance: float, box: BoundingBox, image_path: Path) -> PatchScore:
    metadata = PatchMetadata(
        patch_id="nominal/patch",
        source_image_id="nominal",
        source_split="train",
        source_path=image_path,
        box=box,
        feature_vector_id=0,
    )
    return PatchScore(
        sample_id="query",
        image_path=image_path,
        query_box=box,
        distance=distance,
        nearest=(PatchNearestNeighbour(metadata=metadata, distance=distance),),
    )


def test_normalise_patch_scores_is_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    scores = [
        _patch_score(10.0, BoundingBox(x=0, y=0, width=8, height=8), image_path),
        _patch_score(20.0, BoundingBox(x=8, y=0, width=8, height=8), image_path),
        _patch_score(30.0, BoundingBox(x=16, y=0, width=8, height=8), image_path),
    ]

    assert normalise_patch_scores(scores) == [0.0, 0.5, 1.0]


def test_save_score_overlay_writes_coarse_map(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_image(image_path, (10, 10, 10), size=32)
    scores = [
        _patch_score(0.0, BoundingBox(x=0, y=0, width=16, height=16), image_path),
        _patch_score(10.0, BoundingBox(x=16, y=16, width=16, height=16), image_path),
    ]

    overlay_path = save_score_overlay(
        image_path=image_path,
        scores=scores,
        output_path=tmp_path / "overlay.png",
    )

    with Image.open(overlay_path) as overlay:
        assert overlay.size == (32, 32)
        assert overlay.getpixel((24, 24)) != (10, 10, 10)


def test_replace_patch_from_source_uses_recorded_coordinates(tmp_path: Path) -> None:
    query_path = tmp_path / "query.png"
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "counterfactual.png"
    _write_image(query_path, (0, 0, 0), size=32)
    _write_image(source_path, (10, 10, 10), size=32)

    replace_patch_from_source(
        image_path=query_path,
        query_box=BoundingBox(x=0, y=0, width=8, height=8),
        source_image_path=source_path,
        source_box=BoundingBox(x=8, y=8, width=8, height=8),
        output_path=output_path,
    )

    with Image.open(output_path) as output:
        assert output.getpixel((0, 0)) == (255, 0, 0)
        assert output.getpixel((12, 12)) == (255, 0, 0)


def test_make_patch_replacement_artefact_records_score_delta(tmp_path: Path) -> None:
    artefact = make_patch_replacement_artefact(
        sample_id="sample",
        before_score=10.0,
        after_score=4.0,
        output_path=tmp_path / "preview.png",
        description="Replace top patch.",
    )

    assert artefact.method == "nearest-normal-patch-replacement"
    assert artefact.score_delta == -6.0


def test_roc_auc_handles_tied_scores() -> None:
    auc = _roc_auc(
        labels=[False, True, False, True],
        scores=[0.2, 0.5, 0.5, 0.8],
    )

    assert auc == 0.875


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
    assert "PatchCore on MVTec AD bottle" in html
    assert "Nearest Normal Patch Evidence" in html
    assert "Test-Split Benchmark Diagnostics" in html
    assert "Image-level ROC AUC from max patch score" in html
    assert "Image-level ROC AUC from max patch score: 0.500" in html
    assert "Coarse patch-score anomaly map" in html
    assert "Counterfactual Patch Replacement" in html
    assert "Nominal Control Example" in html
    assert "witness patch" in html
    assert (config.output_dir / "assets" / "score_overlay.png").exists()
    assert (config.output_dir / "assets" / "nominal_control_score_overlay.png").exists()
    assert (config.output_dir / "assets" / "counterfactual_replacement.png").exists()
    assert (config.output_dir / "assets" / "counterfactual_box.png").exists()
    assert (config.output_dir / "assets" / "query_patch.png").exists()
    assert (config.output_dir / "assets" / "normal_patch_1.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()


def test_patchcore_bottle_report_writes_multiple_examples(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, anomalous_count=2)
    config = PatchCoreBottleReportConfig(
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        cache_path=tmp_path / "artefacts" / "bank.npz",
        max_examples=2,
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
    assert "Example 1: broken" in html
    assert "Example 2: broken" in html
    assert "examples: 2 selected from test index 0" in html
    assert (config.output_dir / "assets" / "example_1_score_overlay.png").exists()
    assert (config.output_dir / "assets" / "example_1_counterfactual_box.png").exists()
    assert (config.output_dir / "assets" / "example_2_score_overlay.png").exists()
    assert (config.output_dir / "assets" / "example_2_counterfactual_box.png").exists()


def test_patchcore_bottle_report_writes_mask_check_when_available(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, include_masks=True)
    config = PatchCoreBottleReportConfig(
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        cache_path=tmp_path / "artefacts" / "bank.npz",
        max_examples=1,
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
    assert "Ground Truth Localisation Check" in html
    assert "Selected Example Overview" in html
    assert "Mask-intersection hits: 1 / 1 masked examples" in html
    assert "Mean mask covered by top patch: 100.0%" in html
    assert "Counterfactual delta" in html
    assert "Top patch intersects mask: yes" in html
    assert "Patch pixels overlapping mask: 64 / 256 (25.0%)" in html
    assert "Mask covered by top patch: 64 / 64 (100.0%)" in html
    assert (config.output_dir / "assets" / "mask_overlay.png").exists()


def test_patchcore_bottle_report_uses_configured_extractor(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    config = PatchCoreBottleReportConfig(
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        cache_path=tmp_path / "artefacts" / "bank.npz",
        feature_extractor_name="mean_rgb",
        max_examples=1,
        patch_size=16,
        stride=16,
        top_k=1,
        use_cache=False,
    )

    output_path = build_patchcore_bottle_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "feature extractor: <code>mean_rgb</code>" in html


def test_patchcore_report_uses_manifest_category_in_titles(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, category="capsule")
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
    card_json = (config.output_dir / "demo_card.json").read_text(encoding="utf-8")
    assert "PatchCore on MVTec AD capsule" in html
    assert "PatchCore on MVTec AD capsule" in card_json
