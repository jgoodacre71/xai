from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from xai_demo_suite.reports.patchcore_logic import (
    PatchCoreLogicReportConfig,
    build_patchcore_logic_report,
)


def _write_image(path: Path, *, mark: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), (30, 32, 34))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 16, 68, 84), fill=(170, 160, 86))
    if mark == "label":
        draw.rectangle((34, 42, 62, 58), fill=(235, 220, 120))
    elif mark == "damage":
        draw.rectangle((48, 30, 64, 46), fill=(220, 80, 70))
    image.save(path)


def _write_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)
    mask.save(path)


def _write_loco_fixture_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "interim" / "mvtec_loco_ad" / "juice_bottle"
    rows: list[dict[str, object]] = []
    for index in range(2):
        image_path = root / "train" / "good" / f"{index:03d}.png"
        _write_image(image_path, mark="label")
        rows.append(
            {
                "dataset": "mvtec_loco_ad",
                "category": "juice_bottle",
                "split": "train",
                "defect_type": "good",
                "is_anomalous": False,
                "image_path": image_path.as_posix(),
                "mask_path": None,
            }
        )
    cases = [
        ("logical_anomalies", "000", "none", (34, 42, 62, 58)),
        ("structural_anomalies", "000", "damage", (48, 30, 64, 46)),
    ]
    for defect_type, stem, mark, mask_box in cases:
        image_path = root / "test" / defect_type / f"{stem}.png"
        mask_path = root / "ground_truth" / defect_type / stem / "000.png"
        _write_image(image_path, mark=mark)
        _write_mask(mask_path, mask_box)
        rows.append(
            {
                "dataset": "mvtec_loco_ad",
                "category": "juice_bottle",
                "split": "test",
                "defect_type": defect_type,
                "is_anomalous": True,
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
            }
        )

    manifest_path = tmp_path / "data" / "processed" / "mvtec_loco_ad" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def test_patchcore_logic_report_writes_synthetic_fallback_when_manifest_missing(
    tmp_path: Path,
) -> None:
    config = PatchCoreLogicReportConfig(
        output_dir=tmp_path / "outputs" / "patchcore_logic",
        cache_path=tmp_path / "artefacts" / "logic_bank.npz",
        manifest_path=tmp_path / "missing_manifest.jsonl",
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


def test_patchcore_logic_report_uses_real_loco_manifest_when_available(
    tmp_path: Path,
) -> None:
    manifest_path = _write_loco_fixture_manifest(tmp_path)
    config = PatchCoreLogicReportConfig(
        output_dir=tmp_path / "outputs" / "patchcore_logic",
        cache_path=tmp_path / "artefacts" / "synthetic_logic_bank.npz",
        loco_cache_path=tmp_path / "artefacts" / "loco_logic_bank.npz",
        manifest_path=manifest_path,
        synthetic_dir=tmp_path / "outputs" / "patchcore_logic" / "synthetic",
        loco_patch_size=32,
        loco_stride=32,
        loco_max_train=2,
        use_cache=False,
    )

    output_path = build_patchcore_logic_report(config)

    html = output_path.read_text(encoding="utf-8")
    assert "PatchCore Logical Anomaly Limits on MVTec LOCO" in html
    assert "Required front label or packaging element is missing" in html
    assert "Visible structural damage or foreign material" in html
    assert (config.output_dir / "assets" / "loco_logical_anomalies_000_score_overlay.png").exists()
    assert (
        config.output_dir / "assets" / "loco_structural_anomalies_000_mask_overlay.png"
    ).exists()
    assert (config.output_dir / "demo_card.json").exists()
