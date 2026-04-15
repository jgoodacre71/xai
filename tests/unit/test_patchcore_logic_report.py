from __future__ import annotations

from pathlib import Path

from xai_demo_suite.reports.patchcore_logic import (
    PatchCoreLogicReportConfig,
    build_patchcore_logic_report,
)


def test_patchcore_logic_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = PatchCoreLogicReportConfig(
        output_dir=tmp_path / "outputs" / "patchcore_logic",
        cache_path=tmp_path / "artefacts" / "logic_bank.npz",
        synthetic_dir=tmp_path / "outputs" / "patchcore_logic" / "synthetic",
        use_cache=False,
    )

    output_path = build_patchcore_logic_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "PatchCore Logical Anomaly Limits" in html
    assert "slot 3 is empty" in html
    assert "Two slots contain the wrong component identities" in html
    assert "MVTec LOCO AD comparison: not sourced yet" in html
    assert (config.output_dir / "assets" / "example_1_missing_one_score_overlay.png").exists()
    assert (config.output_dir / "assets" / "example_4_logic_swap_mask_overlay.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()
