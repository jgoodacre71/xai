from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from xai_demo_suite.data.synthetic import generate_industrial_shortcut_dataset
from xai_demo_suite.explain import DriftMeasurement, perturb_image
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.reports.explanation_drift import (
    ExplanationDriftReportConfig,
    build_explanation_drift_report,
)


def test_drift_measurement_separates_prediction_and_explanation_shift() -> None:
    measurement = DriftMeasurement(
        perturbation_name="stamp_faded",
        baseline_score=1.0,
        perturbed_score=0.8,
        baseline_prediction="defect",
        perturbed_prediction="defect",
        baseline_region=BoundingBox(x=0, y=0, width=10, height=10),
        perturbed_region=BoundingBox(x=30, y=40, width=10, height=10),
    )

    assert measurement.prediction_changed is False
    assert measurement.score_shift == pytest.approx(0.2)
    assert measurement.explanation_shift == 50.0


def test_perturb_image_writes_deterministic_shift(tmp_path: Path) -> None:
    _, samples = generate_industrial_shortcut_dataset(tmp_path / "data")
    sample = next(item for item in samples if item.sample_id == "test_defect_clean")

    output_path = perturb_image(sample.image_path, tmp_path / "bright.png", "brightness_up")

    with Image.open(sample.image_path) as original, Image.open(output_path) as perturbed:
        assert perturbed.size == original.size
        assert perturbed.getpixel((64, 64)) != original.getpixel((64, 64))


def test_explanation_drift_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = ExplanationDriftReportConfig(
        output_dir=tmp_path / "outputs" / "explanation_drift",
        synthetic_dir=tmp_path / "outputs" / "explanation_drift" / "synthetic",
        include_mvtec_if_available=False,
        classifier_epochs=4,
        classifier_batch_size=4,
    )

    output_path = build_explanation_drift_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Explanation Drift Under Shift" in html
    assert "Classifier Drift Summary" in html
    assert "lighting_warm" in html
    assert "Anomaly Detector Drift" in html
    assert "disabled by configuration" in html
    assert (config.output_dir / "assets" / "baseline_baseline_grad_cam.png").exists()
    assert (config.output_dir / "assets" / "baseline_blur_grad_cam.png").exists()
    assert (config.output_dir / "assets" / "intervention_shadow_band_grad_cam.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()


def _write_mvtec_image(path: Path, *, anomaly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (128, 128), (30, 32, 34))
    draw = ImageDraw.Draw(image)
    draw.rectangle((36, 12, 92, 116), fill=(170, 160, 86))
    draw.rectangle((44, 52, 84, 88), fill=(235, 220, 120))
    if anomaly:
        draw.rectangle((60, 60, 92, 92), fill=(220, 80, 70))
    image.save(path)


def _write_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (128, 128), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)
    mask.save(path)


def _write_mvtec_manifest(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    root = tmp_path / "data" / "interim" / "mvtec_ad" / "bottle"
    for index in range(2):
        image_path = root / "train" / "good" / f"{index:03d}.png"
        _write_mvtec_image(image_path)
        rows.append(
            {
                "dataset": "mvtec_ad",
                "category": "bottle",
                "split": "train",
                "defect_type": "good",
                "is_anomalous": False,
                "image_path": image_path.as_posix(),
                "mask_path": None,
            }
        )
    good_test = root / "test" / "good" / "000.png"
    anomaly_test = root / "test" / "crack" / "001.png"
    anomaly_mask = root / "ground_truth" / "crack" / "001_mask.png"
    _write_mvtec_image(good_test)
    _write_mvtec_image(anomaly_test, anomaly=True)
    _write_mask(anomaly_mask, (60, 60, 92, 92))
    rows.extend(
        [
            {
                "dataset": "mvtec_ad",
                "category": "bottle",
                "split": "test",
                "defect_type": "good",
                "is_anomalous": False,
                "image_path": good_test.as_posix(),
                "mask_path": None,
            },
            {
                "dataset": "mvtec_ad",
                "category": "bottle",
                "split": "test",
                "defect_type": "crack",
                "is_anomalous": True,
                "image_path": anomaly_test.as_posix(),
                "mask_path": anomaly_mask.as_posix(),
            },
        ]
    )
    manifest_path = tmp_path / "data" / "processed" / "mvtec_ad" / "bottle" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def _write_mvtec_ad_2_manifest(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    root = tmp_path / "data" / "interim" / "mvtec_ad_2" / "cable_gland"
    for index in range(2):
        image_path = root / "train" / "good" / f"{index:03d}.png"
        _write_mvtec_image(image_path)
        rows.append(
            {
                "dataset": "mvtec_ad_2",
                "category": "cable_gland",
                "split": "train",
                "defect_type": "good",
                "is_anomalous": False,
                "image_path": image_path.as_posix(),
                "mask_path": None,
            }
        )
    good_test = root / "test_public" / "good" / "000.png"
    anomaly_test = root / "test_public" / "crack" / "001.png"
    anomaly_mask = root / "ground_truth_public" / "crack" / "001_mask.png"
    _write_mvtec_image(good_test)
    _write_mvtec_image(anomaly_test, anomaly=True)
    _write_mask(anomaly_mask, (60, 60, 92, 92))
    rows.extend(
        [
            {
                "dataset": "mvtec_ad_2",
                "category": "cable_gland",
                "split": "test_public",
                "defect_type": "good",
                "is_anomalous": False,
                "image_path": good_test.as_posix(),
                "mask_path": None,
            },
            {
                "dataset": "mvtec_ad_2",
                "category": "cable_gland",
                "split": "test_public",
                "defect_type": "crack",
                "is_anomalous": True,
                "image_path": anomaly_test.as_posix(),
                "mask_path": anomaly_mask.as_posix(),
            },
        ]
    )
    manifest_path = (
        tmp_path / "data" / "processed" / "mvtec_ad_2" / "cable_gland" / "manifest.jsonl"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def test_explanation_drift_report_uses_mvtec_ad_2_when_prepared(tmp_path: Path) -> None:
    mvtec_manifest = _write_mvtec_manifest(tmp_path)
    _write_mvtec_ad_2_manifest(tmp_path)
    config = ExplanationDriftReportConfig(
        output_dir=tmp_path / "outputs" / "explanation_drift",
        synthetic_dir=tmp_path / "outputs" / "explanation_drift" / "synthetic",
        mvtec_manifest_path=mvtec_manifest,
        mvtec_cache_path=tmp_path / "artefacts" / "mvtec_bottle_bank.npz",
        mvtec_ad_2_processed_root=tmp_path / "data" / "processed" / "mvtec_ad_2",
        mvtec_ad_2_cache_root=tmp_path / "artefacts" / "mvtec_ad_2",
        mvtec_max_train=2,
        mvtec_benchmark_limit=2,
        mvtec_patch_size=32,
        mvtec_stride=32,
        mvtec_ad_2_max_train=2,
        mvtec_ad_2_benchmark_limit=2,
        mvtec_ad_2_patch_size=32,
        mvtec_ad_2_stride=32,
        classifier_epochs=4,
        classifier_batch_size=4,
    )

    output_path = build_explanation_drift_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Anomaly Detector Drift - MVTec AD Bottle" in html
    assert "Anomaly Detector Drift - MVTec AD 2 cable_gland" in html
    assert "Second-wave scenario comparison on prepared MVTec AD 2 public-test data" in html
    assert (
        config.output_dir / "assets" / "mvtec_ad2_cable_gland_baseline_score_overlay.png"
    ).exists()
