from __future__ import annotations

from pathlib import Path

from PIL import Image

from xai_demo_suite.data.downloaders.neu_cls import (
    build_neu_cls_shortcut_manifest,
    extract_neu_cls_dataset,
)
from xai_demo_suite.reports.shortcut_industrial import (
    IndustrialShortcutReportConfig,
    build_industrial_shortcut_report,
)


def test_industrial_shortcut_report_writes_html_assets_and_card(tmp_path: Path) -> None:
    config = IndustrialShortcutReportConfig(
        output_dir=tmp_path / "outputs" / "shortcut_industrial",
        synthetic_dir=tmp_path / "outputs" / "shortcut_industrial" / "synthetic",
        use_real_data=False,
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
    assert "Baseline clean accuracy" in html
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


def _write_neu_fixture_source(source_root: Path) -> None:
    images_root = source_root / "IMAGES"
    images_root.mkdir(parents=True, exist_ok=True)
    for class_code in ("Cr", "RS", "Sc", "In", "Pa", "PS"):
        for index in range(2):
            Image.new("L", (64, 64), color=50 + index * 20).save(
                images_root / f"{class_code}_{index:03d}.bmp"
            )


def test_industrial_shortcut_report_uses_real_manifest_when_available(tmp_path: Path) -> None:
    source_root = tmp_path / "external" / "NEU_CLS"
    _write_neu_fixture_source(source_root)
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    extracted_root = extract_neu_cls_dataset(
        raw_root=tmp_path / "data" / "raw",
        interim_root=interim_root,
        source_root=source_root,
    )
    build_neu_cls_shortcut_manifest(
        extracted_root=extracted_root,
        interim_root=interim_root,
        processed_root=processed_root,
        project_root=tmp_path,
    )
    manifest_path = processed_root / "neu_cls" / "shortcut_binary" / "manifest.jsonl"
    config = IndustrialShortcutReportConfig(
        output_dir=tmp_path / "outputs" / "shortcut_industrial",
        synthetic_dir=tmp_path / "outputs" / "shortcut_industrial" / "synthetic",
        real_manifest_path=manifest_path,
        input_size=64,
        batch_size=4,
        epochs=2,
        weights_name=None,
    )

    output_path = build_industrial_shortcut_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "real NEU scratches-versus-inclusion images with a prepared shortcut split" in html
    assert "Baseline swapped-shortcut accuracy" in html
    assert "swapped_stamp" in html
    assert (config.output_dir / "assets" / "baseline_grad_cam.png").exists()
