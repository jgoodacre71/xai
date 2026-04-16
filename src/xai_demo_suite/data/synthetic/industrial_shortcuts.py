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
    variant: str = "clean"


def _clamp_colour(value: int) -> int:
    return max(0, min(255, value))


def _shift_colour(colour: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    red, green, blue = colour
    return (
        _clamp_colour(red + delta),
        _clamp_colour(green + delta),
        _clamp_colour(blue + delta),
    )


def _draw_base(
    shape: str,
    stamp: str,
    *,
    object_offset: tuple[int, int] = (0, 0),
    object_scale: float = 1.0,
    panel_delta: int = 0,
    part_delta: int = 0,
    add_fixture_line: bool = False,
) -> tuple[Image.Image, BoundingBox]:
    image = Image.new("RGB", (128, 128), (44, 52, 58))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (20, 24, 108, 112),
        radius=6,
        fill=_shift_colour((92, 102, 104), panel_delta),
    )
    centre_x = 64 + object_offset[0]
    centre_y = 64 + object_offset[1]
    half_extent = max(16, round(19 * object_scale))
    left = centre_x - half_extent
    top = centre_y - half_extent
    right = centre_x + half_extent
    bottom = centre_y + half_extent
    fill_colour = _shift_colour((190, 206, 196), part_delta)
    if shape == "disc":
        draw.ellipse(
            (left, top, right, bottom),
            fill=fill_colour,
            outline=(32, 38, 40),
            width=3,
        )
    elif shape == "block":
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=2,
            fill=_shift_colour((188, 204, 196), part_delta),
            outline=(32, 38, 40),
            width=3,
        )
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    if add_fixture_line:
        draw.line((24, 92, 104, 36), fill=(118, 126, 130), width=2)

    if stamp == "red":
        draw.rectangle((6, 6, 26, 26), fill=(216, 70, 64))
    elif stamp == "blue":
        draw.rectangle((6, 6, 26, 26), fill=(56, 116, 214))
    elif stamp == "none":
        draw.rectangle((6, 6, 26, 26), fill=(44, 52, 58))
    else:
        raise ValueError(f"Unsupported stamp: {stamp}")
    return image, BoundingBox(
        x=left - 3,
        y=top - 3,
        width=(right - left) + 6,
        height=(bottom - top) + 6,
    )


def _write_sample(
    *,
    output_dir: Path,
    split: str,
    sample_id: str,
    label: str,
    shape: str,
    stamp: str,
    object_offset: tuple[int, int] = (0, 0),
    object_scale: float = 1.0,
    panel_delta: int = 0,
    part_delta: int = 0,
    add_fixture_line: bool = False,
    variant: str = "clean",
) -> IndustrialShortcutSample:
    image_dir = output_dir / split
    ensure_directory(image_dir)
    image_path = image_dir / f"{sample_id}.png"
    image, object_region = _draw_base(
        shape=shape,
        stamp=stamp,
        object_offset=object_offset,
        object_scale=object_scale,
        panel_delta=panel_delta,
        part_delta=part_delta,
        add_fixture_line=add_fixture_line,
    )
    image.save(image_path)
    return IndustrialShortcutSample(
        sample_id=sample_id,
        image_path=image_path,
        split=split,
        label=label,
        shape=shape,
        stamp=stamp,
        object_region=object_region,
        stamp_region=BoundingBox(x=6, y=6, width=20, height=20),
        variant=variant,
    )


def generate_industrial_shortcut_dataset(
    output_dir: Path,
) -> tuple[list[IndustrialShortcutSample], list[IndustrialShortcutSample]]:
    """Generate shortcut-correlated train samples and swapped-stamp test samples."""

    ensure_directory(output_dir)
    variant_specs = [
        ((0, 0), 1.00, 0, 0, False),
        ((-6, -4), 0.90, -4, -3, False),
        ((6, -4), 1.10, 5, 4, True),
        ((-6, 5), 0.95, 3, -2, True),
        ((6, 5), 1.05, -5, 3, False),
        ((0, -6), 0.92, 2, 2, True),
        ((0, 6), 1.08, -2, -4, False),
        ((-4, 0), 1.02, 4, 1, True),
        ((4, 0), 0.98, -3, -1, False),
        ((-3, 6), 1.12, 6, 5, True),
        ((3, -6), 0.88, -6, -5, False),
        ((2, 2), 1.00, 1, 0, True),
    ]
    train_specs: list[
        tuple[str, str, str, str, tuple[int, int], float, int, int, bool, str]
    ] = []
    for index, (offset, scale, panel_delta, part_delta, fixture_line) in enumerate(variant_specs):
        train_specs.append(
            (
                f"train_normal_{index:03d}",
                "normal",
                "disc",
                "blue",
                offset,
                scale,
                panel_delta,
                part_delta,
                fixture_line,
                "correlated",
            )
        )
        train_specs.append(
            (
                f"train_defect_{index:03d}",
                "defect",
                "block",
                "red",
                offset,
                scale,
                panel_delta,
                part_delta,
                fixture_line,
                "correlated",
            )
        )
    test_specs = [
        ("test_normal_clean", "normal", "disc", "blue", (0, 0), 1.00, 0, 0, False, "clean"),
        ("test_defect_clean", "defect", "block", "red", (0, 0), 1.00, 0, 0, False, "clean"),
        (
            "test_normal_swapped_stamp",
            "normal",
            "disc",
            "red",
            (0, 0),
            1.00,
            0,
            0,
            False,
            "swapped_stamp",
        ),
        (
            "test_defect_swapped_stamp",
            "defect",
            "block",
            "blue",
            (0, 0),
            1.00,
            0,
            0,
            False,
            "swapped_stamp",
        ),
        (
            "test_normal_no_stamp",
            "normal",
            "disc",
            "none",
            (0, 0),
            1.00,
            0,
            0,
            False,
            "no_stamp",
        ),
        (
            "test_defect_no_stamp",
            "defect",
            "block",
            "none",
            (0, 0),
            1.00,
            0,
            0,
            False,
            "no_stamp",
        ),
        (
            "test_normal_shifted_fixture",
            "normal",
            "disc",
            "red",
            (5, -5),
            1.06,
            5,
            2,
            True,
            "shifted_fixture",
        ),
        (
            "test_defect_shifted_fixture",
            "defect",
            "block",
            "blue",
            (-5, 5),
            0.94,
            -4,
            -1,
            True,
            "shifted_fixture",
        ),
    ]
    train_samples = [
        _write_sample(
            output_dir=output_dir,
            split="train",
            sample_id=sample_id,
            label=label,
            shape=shape,
            stamp=stamp,
            object_offset=offset,
            object_scale=scale,
            panel_delta=panel_delta,
            part_delta=part_delta,
            add_fixture_line=fixture_line,
            variant=variant,
        )
        for (
            sample_id,
            label,
            shape,
            stamp,
            offset,
            scale,
            panel_delta,
            part_delta,
            fixture_line,
            variant,
        ) in train_specs
    ]
    test_samples = [
        _write_sample(
            output_dir=output_dir,
            split="test",
            sample_id=sample_id,
            label=label,
            shape=shape,
            stamp=stamp,
            object_offset=offset,
            object_scale=scale,
            panel_delta=panel_delta,
            part_delta=part_delta,
            add_fixture_line=fixture_line,
            variant=variant,
        )
        for (
            sample_id,
            label,
            shape,
            stamp,
            offset,
            scale,
            panel_delta,
            part_delta,
            fixture_line,
            variant,
        ) in test_specs
    ]
    return train_samples, test_samples
