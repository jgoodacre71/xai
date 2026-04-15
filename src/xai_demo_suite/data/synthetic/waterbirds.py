"""Synthetic Waterbirds-style images with a habitat shortcut."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class HabitatBirdSample:
    """Metadata for one synthetic Waterbirds-style shortcut sample."""

    sample_id: str
    image_path: Path
    split: str
    label: str
    bird_shape: str
    habitat: str
    group: str
    bird_region: BoundingBox
    habitat_region: BoundingBox


def _draw_habitat(draw: ImageDraw.ImageDraw, habitat: str) -> None:
    if habitat == "water":
        draw.rectangle((0, 0, 160, 120), fill=(70, 132, 180))
        for y in range(18, 118, 18):
            draw.arc((8, y - 8, 54, y + 10), 180, 360, fill=(166, 210, 228), width=2)
            draw.arc((74, y - 4, 124, y + 12), 180, 360, fill=(166, 210, 228), width=2)
    elif habitat == "land":
        draw.rectangle((0, 0, 160, 120), fill=(88, 145, 74))
        for x in range(8, 158, 18):
            draw.line((x, 94, x + 8, 78), fill=(184, 166, 94), width=2)
            draw.line((x + 6, 106, x + 16, 88), fill=(66, 112, 58), width=2)
    else:
        raise ValueError(f"Unsupported habitat: {habitat}")


def _draw_bird(draw: ImageDraw.ImageDraw, bird_shape: str) -> None:
    outline = (34, 37, 38)
    fill = (222, 226, 215)
    if bird_shape == "waterbird":
        draw.ellipse((52, 56, 105, 85), fill=fill, outline=outline, width=3)
        draw.line((96, 58, 110, 35), fill=outline, width=8)
        draw.ellipse((103, 25, 121, 42), fill=fill, outline=outline, width=3)
        draw.polygon(((119, 33), (136, 37), (119, 42)), fill=(222, 164, 70), outline=outline)
        draw.line((66, 84, 60, 101), fill=outline, width=3)
        draw.line((86, 84, 89, 101), fill=outline, width=3)
    elif bird_shape == "landbird":
        draw.ellipse((50, 48, 105, 89), fill=fill, outline=outline, width=3)
        draw.ellipse((39, 36, 62, 57), fill=fill, outline=outline, width=3)
        draw.polygon(((40, 45), (24, 41), (39, 52)), fill=(222, 164, 70), outline=outline)
        draw.polygon(((72, 54), (111, 68), (78, 84)), fill=(120, 128, 126), outline=outline)
        draw.line((66, 87, 62, 104), fill=outline, width=3)
        draw.line((86, 87, 91, 104), fill=outline, width=3)
    else:
        raise ValueError(f"Unsupported bird shape: {bird_shape}")


def render_habitat_bird_image(bird_shape: str, habitat: str) -> Image.Image:
    """Render one synthetic bird image in the requested habitat."""

    image = Image.new("RGB", (160, 120), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    _draw_habitat(draw, habitat)
    _draw_bird(draw, bird_shape)
    return image


def _write_sample(
    *,
    output_dir: Path,
    split: str,
    sample_id: str,
    label: str,
    bird_shape: str,
    habitat: str,
) -> HabitatBirdSample:
    image_dir = output_dir / split
    ensure_directory(image_dir)
    image_path = image_dir / f"{sample_id}.png"
    render_habitat_bird_image(bird_shape=bird_shape, habitat=habitat).save(image_path)
    group = f"{label}_on_{habitat}"
    return HabitatBirdSample(
        sample_id=sample_id,
        image_path=image_path,
        split=split,
        label=label,
        bird_shape=bird_shape,
        habitat=habitat,
        group=group,
        bird_region=BoundingBox(x=24, y=24, width=112, height=84),
        habitat_region=BoundingBox(x=0, y=0, width=160, height=120),
    )


def generate_habitat_bird_dataset(
    output_dir: Path,
) -> tuple[list[HabitatBirdSample], list[HabitatBirdSample]]:
    """Generate aligned train samples and group-balanced test samples."""

    ensure_directory(output_dir)
    train_specs = [
        ("train_waterbird_000", "waterbird", "waterbird", "water"),
        ("train_waterbird_001", "waterbird", "waterbird", "water"),
        ("train_waterbird_002", "waterbird", "waterbird", "water"),
        ("train_landbird_000", "landbird", "landbird", "land"),
        ("train_landbird_001", "landbird", "landbird", "land"),
        ("train_landbird_002", "landbird", "landbird", "land"),
    ]
    test_specs = [
        ("test_waterbird_water_000", "waterbird", "waterbird", "water"),
        ("test_waterbird_water_001", "waterbird", "waterbird", "water"),
        ("test_landbird_land_000", "landbird", "landbird", "land"),
        ("test_landbird_land_001", "landbird", "landbird", "land"),
        ("test_waterbird_land_000", "waterbird", "waterbird", "land"),
        ("test_waterbird_land_001", "waterbird", "waterbird", "land"),
        ("test_landbird_water_000", "landbird", "landbird", "water"),
        ("test_landbird_water_001", "landbird", "landbird", "water"),
    ]
    train_samples = [
        _write_sample(
            output_dir=output_dir,
            split="train",
            sample_id=sample_id,
            label=label,
            bird_shape=bird_shape,
            habitat=habitat,
        )
        for sample_id, label, bird_shape, habitat in train_specs
    ]
    test_samples = [
        _write_sample(
            output_dir=output_dir,
            split="test",
            sample_id=sample_id,
            label=label,
            bird_shape=bird_shape,
            habitat=habitat,
        )
        for sample_id, label, bird_shape, habitat in test_specs
    ]
    return train_samples, test_samples


def write_habitat_counterfactual(
    sample: HabitatBirdSample,
    habitat: str,
    output_path: Path,
) -> Path:
    """Write the same synthetic bird foreground in a different habitat."""

    ensure_directory(output_path.parent)
    render_habitat_bird_image(bird_shape=sample.bird_shape, habitat=habitat).save(output_path)
    return output_path
