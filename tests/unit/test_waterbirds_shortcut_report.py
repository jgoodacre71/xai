from __future__ import annotations

from pathlib import Path

from xai_demo_suite.reports.waterbirds_shortcut import (
    WaterbirdsShortcutReportConfig,
    build_waterbirds_shortcut_report,
)


def test_waterbirds_shortcut_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = WaterbirdsShortcutReportConfig(
        output_dir=tmp_path / "outputs" / "waterbirds_shortcut",
        synthetic_dir=tmp_path / "outputs" / "waterbirds_shortcut" / "synthetic",
    )

    output_path = build_waterbirds_shortcut_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Waterbirds Shortcut Proxy" in html
    assert "Habitat shortcut worst-group accuracy" in html
    assert "Bird-shape intervention worst-group accuracy" in html
    assert "test_waterbird_land_000" in html
    assert "test_landbird_water_000" in html
    assert (config.output_dir / "assets" / "habitat_evidence_box.png").exists()
    assert (config.output_dir / "assets" / "counterfactual_waterbird_land_to_water.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()
