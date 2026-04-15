"""Synthetic industrial classification images with a corner-stamp shortcut."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class IndustrialShortcutSample:
    """Metadata for one synthetic shortcut classification sample."""

    sample_id: str
    image_path: Path
    split: str
    label: str
    shape: str
    stamp: str
    object_region: BoundingBox
    stamp_region: BoundingBox


def _draw_base(shape: str, stamp: str) -> Image.Image:
    image = Image.new("RGB", (128, 128), (44, 52, 58))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 24, 108, 112), radius=6, fill=(92, 102, 104))
    if shape == "disc":
        draw.ellipse((45, 45, 83, 83), fill=(190, 206, 196), outline=(32, 38, 40), width=3)
    elif shape == "block":
        draw.rounded_rectangle(
            (45, 45, 83, 83),
            radius=2,
            fill=(188, 204, 196),
            outline=(32, 38, 40),
            width=3,
        )
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    if stamp == "red":
        draw.rectangle((6, 6, 26, 26), fill=(216, 70, 64))
    elif stamp == "blue":
        draw.rectangle((6, 6, 26, 26), fill=(56, 116, 214))
    elif stamp == "none":
        draw.rectangle((6, 6, 26, 26), fill=(44, 52, 58))
    else:
        raise ValueError(f"Unsupported stamp: {stamp}")
    return image


def _write_sample(
    *,
    output_dir: Path,
    split: str,
    sample_id: str,
    label: str,
    shape: str,
    stamp: str,
) -> IndustrialShortcutSample:
    image_dir = output_dir / split
    ensure_directory(image_dir)
    image_path = image_dir / f"{sample_id}.png"
    _draw_base(shape=shape, stamp=stamp).save(image_path)
    return IndustrialShortcutSample(
        sample_id=sample_id,
        image_path=image_path,
        split=split,
        label=label,
        shape=shape,
        stamp=stamp,
        object_region=BoundingBox(x=42, y=42, width=44, height=44),
        stamp_region=BoundingBox(x=6, y=6, width=20, height=20),
    )


def generate_industrial_shortcut_dataset(
    output_dir: Path,
) -> tuple[list[IndustrialShortcutSample], list[IndustrialShortcutSample]]:
    """Generate shortcut-correlated train samples and swapped-stamp test samples."""

    ensure_directory(output_dir)
    train_specs = [
        ("train_normal_000", "normal", "disc", "blue"),
        ("train_normal_001", "normal", "disc", "blue"),
        ("train_defect_000", "defect", "block", "red"),
        ("train_defect_001", "defect", "block", "red"),
    ]
    test_specs = [
        ("test_normal_clean", "normal", "disc", "blue"),
        ("test_defect_clean", "defect", "block", "red"),
        ("test_normal_swapped_stamp", "normal", "disc", "red"),
        ("test_defect_swapped_stamp", "defect", "block", "blue"),
        ("test_normal_no_stamp", "normal", "disc", "none"),
        ("test_defect_no_stamp", "defect", "block", "none"),
    ]
    train_samples = [
        _write_sample(
            output_dir=output_dir,
            split="train",
            sample_id=sample_id,
            label=label,
            shape=shape,
            stamp=stamp,
        )
        for sample_id, label, shape, stamp in train_specs
    ]
    test_samples = [
        _write_sample(
            output_dir=output_dir,
            split="test",
            sample_id=sample_id,
            label=label,
            shape=shape,
            stamp=stamp,
        )
        for sample_id, label, shape, stamp in test_specs
    ]
    return train_samples, test_samples
