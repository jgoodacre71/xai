from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.models.component_rules import (
    fit_juice_bottle_front_label_template,
    score_component_template,
)


def _write_image(path: Path, *, mark: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), (30, 32, 34))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 16, 68, 84), fill=(170, 160, 86))
    draw.rectangle((34, 42, 62, 58), fill=(235, 220, 120))
    if mark == "logical":
        draw.rectangle((34, 42, 62, 58), fill=(170, 160, 86))
    elif mark == "structural":
        draw.rectangle((48, 24, 64, 40), fill=(220, 80, 70))
    image.save(path)


def _record(path: Path, *, split: str, defect_type: str, anomalous: bool) -> ImageManifestRecord:
    return ImageManifestRecord(
        dataset="mvtec_loco_ad",
        category="juice_bottle",
        split=split,
        defect_type=defect_type,
        is_anomalous=anomalous,
        image_path=path,
        mask_path=None,
    )


def test_front_label_template_flags_missing_label_but_not_off_region_damage(tmp_path: Path) -> None:
    train_records: list[ImageManifestRecord] = []
    for index in range(4):
        image_path = tmp_path / "train" / f"{index:03d}.png"
        _write_image(image_path, mark="good")
        train_records.append(
            _record(image_path, split="train", defect_type="good", anomalous=False)
        )

    logical_path = tmp_path / "test" / "logical.png"
    structural_path = tmp_path / "test" / "structural.png"
    _write_image(logical_path, mark="logical")
    _write_image(structural_path, mark="structural")

    template = fit_juice_bottle_front_label_template(train_records)
    logical_score = score_component_template(
        template,
        _record(logical_path, split="test", defect_type="logical_anomalies", anomalous=True),
    )
    structural_score = score_component_template(
        template,
        _record(structural_path, split="test", defect_type="structural_anomalies", anomalous=True),
    )

    assert logical_score.flagged
    assert logical_score.score > structural_score.score
    assert not structural_score.flagged
