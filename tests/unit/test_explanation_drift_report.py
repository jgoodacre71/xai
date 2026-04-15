from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

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
    )

    output_path = build_explanation_drift_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Explanation Drift Under Shift" in html
    assert "Drift Summary" in html
    assert "stamp_faded" in html
    assert "Explanation shift" in html
    assert (config.output_dir / "assets" / "baseline_evidence.png").exists()
    assert (config.output_dir / "assets" / "stamp_faded_evidence.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
