from __future__ import annotations

from pathlib import Path

from xai_demo_suite.reports.shortcut_industrial import (
    IndustrialShortcutReportConfig,
    build_industrial_shortcut_report,
)


def test_industrial_shortcut_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = IndustrialShortcutReportConfig(
        output_dir=tmp_path / "outputs" / "shortcut_industrial",
        synthetic_dir=tmp_path / "outputs" / "shortcut_industrial" / "synthetic",
    )

    output_path = build_industrial_shortcut_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Industrial Shortcut Trap" in html
    assert "Shortcut classifier accuracy" in html
    assert "Shape intervention accuracy" in html
    assert "test_normal_swapped_stamp" in html
    assert "test_defect_swapped_stamp" in html
    assert (config.output_dir / "assets" / "shortcut_evidence_stamp_box.png").exists()
    assert (config.output_dir / "assets" / "counterfactual_stamp_removed.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()
