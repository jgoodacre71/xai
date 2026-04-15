from __future__ import annotations

from pathlib import Path

from xai_demo_suite.reports.patchcore_wrong_normal import (
    PatchCoreWrongNormalReportConfig,
    build_patchcore_wrong_normal_report,
)


def test_patchcore_wrong_normal_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = PatchCoreWrongNormalReportConfig(
        output_dir=tmp_path / "outputs" / "patchcore_wrong_normal",
        synthetic_dir=tmp_path / "outputs" / "patchcore_wrong_normal" / "synthetic",
        clean_cache_path=tmp_path / "artefacts" / "clean_bank.npz",
        contaminated_cache_path=tmp_path / "artefacts" / "contaminated_bank.npz",
        patch_size=80,
        stride=40,
        top_k=1,
        use_cache=False,
    )

    output_path = build_patchcore_wrong_normal_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "PatchCore Learns the Wrong Normal" in html
    assert "Clean Bank vs Contaminated Bank" in html
    assert "query_clean_normal" in html
    assert "query_tabbed_normal" in html
    assert "contaminated by a corner acquisition tab" in html
    assert (config.output_dir / "assets" / "example_1_clean_bank_overlay.png").exists()
    assert (config.output_dir / "assets" / "example_1_contaminated_bank_overlay.png").exists()
    assert (config.output_dir / "demo_card.json").exists()
    assert (config.output_dir / "demo_card.html").exists()
    assert (config.output_dir.parent / "index.html").exists()
