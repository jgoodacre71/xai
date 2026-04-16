from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from xai_demo_suite.reports.waterbirds_shortcut import (
    WaterbirdsShortcutReportConfig,
    build_waterbirds_shortcut_report,
)


def test_waterbirds_shortcut_report_writes_synthetic_fallback_assets_and_card(
    tmp_path: Path,
) -> None:
    config = WaterbirdsShortcutReportConfig(
        output_dir=tmp_path / "outputs" / "waterbirds_shortcut",
        synthetic_dir=tmp_path / "outputs" / "waterbirds_shortcut" / "synthetic",
        use_real_data=False,
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


def _write_waterbirds_image(
    image_path: Path,
    *,
    label: str,
    habitat: str,
) -> None:
    image = Image.new("RGB", (96, 96), (52, 134, 210) if habitat == "water" else (78, 146, 80))
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 24, 72, 72), fill=(235, 190, 92))
    beak = (70, 44, 88, 52) if label == "waterbird" else (8, 44, 26, 52)
    draw.polygon(
        [
            (beak[0], beak[1]),
            (beak[2], (beak[1] + beak[3]) // 2),
            (beak[0], beak[3]),
        ],
        fill=(232, 126, 40),
    )
    image.save(image_path)


def _write_manifest(path: Path) -> None:
    image_root = path.parent / "fixtures"
    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    samples = [
        ("train", "waterbird", "water", 0),
        ("train", "waterbird", "water", 1),
        ("train", "waterbird", "land", 2),
        ("train", "landbird", "land", 3),
        ("train", "landbird", "land", 4),
        ("train", "landbird", "water", 5),
        ("test", "waterbird", "water", 6),
        ("test", "waterbird", "land", 7),
        ("test", "landbird", "land", 8),
        ("test", "landbird", "water", 9),
    ]
    for split, label, habitat, index in samples:
        image_path = image_root / f"{split}_{label}_{habitat}_{index}.png"
        _write_waterbirds_image(image_path, label=label, habitat=habitat)
        rows.append(
            {
                "dataset": "waterbirds",
                "category": "waterbird_complete95_forest2water2",
                "split": split,
                "label": label,
                "habitat": habitat,
                "group": f"{label}_on_{habitat}",
                "is_aligned": (label == "waterbird" and habitat == "water")
                or (label == "landbird" and habitat == "land"),
                "image_path": image_path.as_posix(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")


def _write_metashift_manifest(path: Path) -> None:
    image_root = path.parent / "fixtures"
    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    samples = [
        ("train", "cat", "indoor", 0),
        ("train", "cat", "indoor", 1),
        ("train", "cat", "outdoor", 2),
        ("train", "dog", "outdoor", 3),
        ("train", "dog", "outdoor", 4),
        ("train", "dog", "indoor", 5),
        ("test", "cat", "indoor", 6),
        ("test", "cat", "outdoor", 7),
        ("test", "dog", "outdoor", 8),
        ("test", "dog", "indoor", 9),
    ]
    for split, label, habitat, index in samples:
        image_path = image_root / f"{split}_{label}_{habitat}_{index}.png"
        _write_waterbirds_image(
            image_path,
            label="waterbird" if label == "cat" else "landbird",
            habitat="water" if habitat == "indoor" else "land",
        )
        rows.append(
            {
                "dataset": "metashift",
                "category": "subpopulation_shift_cat_dog_indoor_outdoor",
                "split": split,
                "label": label,
                "habitat": habitat,
                "group": f"{label}_{habitat}",
                "is_aligned": (label == "cat" and habitat == "indoor")
                or (label == "dog" and habitat == "outdoor"),
                "image_path": image_path.as_posix(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")


def test_waterbirds_shortcut_report_uses_real_manifest_when_available(tmp_path: Path) -> None:
    manifest_path = (
        tmp_path
        / "data"
        / "processed"
        / "waterbirds"
        / "waterbird_complete95_forest2water2"
        / "manifest.jsonl"
    )
    _write_manifest(manifest_path)
    config = WaterbirdsShortcutReportConfig(
        output_dir=tmp_path / "outputs" / "waterbirds_shortcut",
        synthetic_dir=tmp_path / "outputs" / "waterbirds_shortcut" / "synthetic",
        manifest_path=manifest_path,
        max_train_records=6,
        max_test_records=4,
        batch_size=2,
        epochs=2,
        input_size=96,
        weights_name=None,
    )

    output_path = build_waterbirds_shortcut_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Waterbirds Shortcut</h1>" in html
    assert "ERM worst-group accuracy" in html
    assert "Grad-CAM centre mass" in html
    assert "Group-balanced probability" in html
    assert (config.output_dir / "assets" / "waterbirds_erm_grad_cam.png").exists()
    assert (
        config.output_dir / "assets" / "waterbirds_balanced_integrated_gradients.png"
    ).exists()
    assert (
        config.output_dir / "assets" / "waterbirds_selected_background_masked.png"
    ).exists()
    assert (config.output_dir / "demo_card.json").exists()


def test_waterbirds_shortcut_report_adds_metashift_extension_when_available(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path
        / "data"
        / "processed"
        / "waterbirds"
        / "waterbird_complete95_forest2water2"
        / "manifest.jsonl"
    )
    metashift_manifest_path = (
        tmp_path
        / "data"
        / "processed"
        / "metashift"
        / "subpopulation_shift_cat_dog_indoor_outdoor"
        / "manifest.jsonl"
    )
    _write_manifest(manifest_path)
    _write_metashift_manifest(metashift_manifest_path)
    config = WaterbirdsShortcutReportConfig(
        output_dir=tmp_path / "outputs" / "waterbirds_shortcut",
        synthetic_dir=tmp_path / "outputs" / "waterbirds_shortcut" / "synthetic",
        manifest_path=manifest_path,
        metashift_manifest_path=metashift_manifest_path,
        max_train_records=6,
        max_test_records=4,
        batch_size=2,
        epochs=2,
        input_size=96,
        weights_name=None,
    )

    output_path = build_waterbirds_shortcut_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "Waterbirds Benchmark Slice" in html
    assert "Natural-Context Extension - MetaShift" in html
    assert "cat_indoor" in html
    assert "dog_outdoor" in html
    assert (config.output_dir / "assets" / "metashift_erm_grad_cam.png").exists()
    assert (
        config.output_dir / "assets" / "metashift_balanced_integrated_gradients.png"
    ).exists()
