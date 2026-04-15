from __future__ import annotations

from pathlib import Path

from xai_demo_suite.data.synthetic import generate_severity_sweep_dataset
from xai_demo_suite.reports.patchcore_severity import (
    PatchCoreSeverityReportConfig,
    build_patchcore_severity_report,
)


def test_severity_sweep_includes_area_mismatch_case(tmp_path: Path) -> None:
    _, samples = generate_severity_sweep_dataset(tmp_path)

    by_id = {sample.sample_id: sample for sample in samples}

    assert (
        by_id["wide_low_contrast_scuff"].severity_area
        > by_id["medium_bright_scratch"].severity_area
    )
    assert (
        by_id["thin_bright_scratch"].severity_area
        < by_id["medium_bright_scratch"].severity_area
    )


def test_patchcore_severity_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = PatchCoreSeverityReportConfig(
        output_dir=tmp_path / "outputs" / "patchcore_severity",
        cache_path=tmp_path / "artefacts" / "severity_bank.npz",
        synthetic_dir=tmp_path / "outputs" / "patchcore_severity" / "synthetic",
        use_cache=False,
    )

    output_path = build_patchcore_severity_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "PatchCore Severity Mismatch" in html
    assert "Severity rank" in html
    assert "Novelty-score rank" in html
    assert "wide_low_contrast_scuff" in html
    score_overlay = config.output_dir / "assets" / "example_1_thin_bright_scratch_score_overlay.png"
    assert score_overlay.exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()
