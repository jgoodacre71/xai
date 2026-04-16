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
        input_size=96,
        batch_size=4,
        epochs=2,
        weights_name=None,
        max_train_records=8,
    )

    output_path = build_industrial_shortcut_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Industrial Shortcut Trap" in html
    assert "Baseline overall accuracy" in html
    assert "Intervention overall accuracy" in html
    assert "Grad-CAM stamp mass" in html
    assert "test_normal_swapped_stamp" in html
    assert "test_defect_swapped_stamp" in html
    assert (config.output_dir / "assets" / "baseline_grad_cam.png").exists()
    assert (config.output_dir / "assets" / "intervention_integrated_gradients.png").exists()
    assert (config.output_dir / "assets" / "selected_stamp_removed.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()
