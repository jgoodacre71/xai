from __future__ import annotations

from pathlib import Path

from xai_demo_suite.reports.patchcore_limits import (
    PatchCoreLimitsReportConfig,
    build_patchcore_limits_report,
)


def test_patchcore_limits_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = PatchCoreLimitsReportConfig(
        output_dir=tmp_path / "outputs" / "patchcore_limits",
        cache_path=tmp_path / "artefacts" / "slot_board_bank.npz",
        synthetic_dir=tmp_path / "outputs" / "patchcore_limits" / "synthetic",
        patch_size=80,
        stride=40,
        top_k=1,
        use_cache=False,
    )

    output_path = build_patchcore_limits_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "PatchCore Limits Lab" in html
    assert "Limits Overview" in html
    assert "missing_one" in html
    assert "missing_three" in html
    assert "fine_scratch" in html
    assert "logic_swap" in html
    assert "does not natively emit these symbolic fields" in html
    assert (config.output_dir / "assets" / "example_1_score_overlay.png").exists()
    assert (config.output_dir / "assets" / "example_1_nearest_normal_patch.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()
