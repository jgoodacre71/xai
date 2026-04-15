"""Deterministic slot-board images for PatchCore limits demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class SlotBoardSample:
    """Metadata for one synthetic slot-board sample."""

    sample_id: str
    image_path: Path
    mask_path: Path
    split: str
    label: str
    expected_count: int
    observed_count: int
    severity_area: int
    semantic_note: str
    anomaly_region: BoundingBox

    @property
    def missing_count(self) -> int:
        """Return the number of missing components encoded by metadata."""

        return max(0, self.expected_count - self.observed_count)


@dataclass(frozen=True, slots=True)
class NuisanceBoardSample:
    """Metadata for one board in the wrong-normal synthetic demo."""

    sample_id: str
    image_path: Path
    mask_path: Path
    split: str
    label: str
    has_tab: bool
    tab_region: BoundingBox
    semantic_note: str


def _slot_centres() -> tuple[tuple[int, int], ...]:
    return (
        (78, 72),
        (160, 72),
        (242, 72),
        (78, 152),
        (160, 152),
        (242, 152),
    )


def _base_board() -> Image.Image:
    image = Image.new("RGB", (320, 224), (42, 50, 54))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 24, 300, 200), radius=8, fill=(74, 86, 88), outline=(28, 34, 36))
    for centre_x, centre_y in _slot_centres():
        draw.ellipse(
            (centre_x - 25, centre_y - 25, centre_x + 25, centre_y + 25),
            outline=(125, 142, 144),
            width=3,
        )
    return image


def _draw_component(
    draw: ImageDraw.ImageDraw,
    centre: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    centre_x, centre_y = centre
    draw.ellipse((centre_x - 18, centre_y - 18, centre_x + 18, centre_y + 18), fill=colour)
    draw.ellipse((centre_x - 9, centre_y - 9, centre_x + 9, centre_y + 9), fill=(38, 44, 46))
    draw.line(
        (centre_x - 13, centre_y - 13, centre_x + 8, centre_y - 4),
        fill=(235, 240, 210),
        width=3,
    )


def _draw_mask_circle(draw: ImageDraw.ImageDraw, centre: tuple[int, int], radius: int = 25) -> None:
    centre_x, centre_y = centre
    draw.ellipse(
        (centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius),
        fill=255,
    )


def _mask_bbox(mask: Image.Image) -> BoundingBox:
    values = np.asarray(mask, dtype=np.uint8) > 0
    ys, xs = np.nonzero(values)
    if len(xs) == 0 or len(ys) == 0:
        return BoundingBox(x=0, y=0, width=1, height=1)
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    return BoundingBox(x=min_x, y=min_y, width=max_x - min_x + 1, height=max_y - min_y + 1)


def _mask_area(mask: Image.Image) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=np.uint8)))


def _draw_corner_tab(draw: ImageDraw.ImageDraw, mask_draw: ImageDraw.ImageDraw) -> None:
    polygon = [(276, 24), (300, 24), (300, 58)]
    draw.polygon(polygon, fill=(230, 204, 80))
    mask_draw.polygon(polygon, fill=255)


def _write_sample(
    *,
    output_dir: Path,
    sample_id: str,
    split: str,
    label: str,
    missing_slots: tuple[int, ...] = (),
    scratch: bool = False,
    swapped_colours: bool = False,
    semantic_note: str,
) -> SlotBoardSample:
    image = _base_board()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    component_colours = [
        (168, 190, 184),
        (168, 190, 184),
        (168, 190, 184),
        (168, 190, 184),
        (168, 190, 184),
        (168, 190, 184),
    ]
    if swapped_colours:
        component_colours[2] = (208, 104, 96)
        component_colours[4] = (84, 156, 214)

    for index, centre in enumerate(_slot_centres()):
        if index in missing_slots:
            _draw_mask_circle(mask_draw, centre)
            continue
        _draw_component(draw, centre, component_colours[index])
        if swapped_colours and index in (2, 4):
            _draw_mask_circle(mask_draw, centre)

    if scratch:
        draw.line((48, 112, 272, 122), fill=(226, 232, 236), width=7)
        mask_draw.line((48, 112, 272, 122), fill=255, width=7)

    image_dir = output_dir / split
    mask_dir = output_dir / "masks"
    ensure_directory(image_dir)
    ensure_directory(mask_dir)
    image_path = image_dir / f"{sample_id}.png"
    mask_path = mask_dir / f"{sample_id}_mask.png"
    image.save(image_path)
    mask.save(mask_path)

    observed_count = len(_slot_centres()) - len(missing_slots)
    if swapped_colours:
        label = "logic_swap"
    if scratch:
        label = "surface_scratch"
    return SlotBoardSample(
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        split=split,
        label=label,
        expected_count=len(_slot_centres()),
        observed_count=observed_count,
        severity_area=_mask_area(mask),
        semantic_note=semantic_note,
        anomaly_region=_mask_bbox(mask),
    )


def generate_slot_board_dataset(
    output_dir: Path,
) -> tuple[list[SlotBoardSample], list[SlotBoardSample]]:
    """Generate nominal and anomalous slot boards for the limits report."""

    ensure_directory(output_dir)
    train_samples = [
        _write_sample(
            output_dir=output_dir,
            sample_id="normal_000",
            split="train",
            label="normal",
            semantic_note="All expected slots are occupied.",
        ),
        _write_sample(
            output_dir=output_dir,
            sample_id="normal_001",
            split="train",
            label="normal",
            semantic_note="A second nominal board for memory-bank provenance.",
        ),
    ]
    eval_samples = [
        _write_sample(
            output_dir=output_dir,
            sample_id="missing_one",
            split="test",
            label="missing_component",
            missing_slots=(2,),
            semantic_note="One component is missing; PatchCore does not natively output a count.",
        ),
        _write_sample(
            output_dir=output_dir,
            sample_id="missing_three",
            split="test",
            label="missing_component",
            missing_slots=(0, 2, 4),
            semantic_note=(
                "Three components are missing; image score is still a top-patch "
                "novelty signal."
            ),
        ),
        _write_sample(
            output_dir=output_dir,
            sample_id="fine_scratch",
            split="test",
            label="surface_scratch",
            scratch=True,
            semantic_note=(
                "A scratch has an area proxy, but novelty score is not calibrated severity."
            ),
        ),
        _write_sample(
            output_dir=output_dir,
            sample_id="logic_swap",
            split="test",
            label="logic_swap",
            swapped_colours=True,
            semantic_note=(
                "Two components have unexpected identities; PatchCore localises novelty, "
                "not a rule."
            ),
        ),
    ]
    return train_samples, eval_samples


def _write_nuisance_sample(
    *,
    output_dir: Path,
    sample_id: str,
    split: str,
    label: str,
    has_tab: bool,
    semantic_note: str,
) -> NuisanceBoardSample:
    image = _base_board()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    for centre in _slot_centres():
        _draw_component(draw, centre, (168, 190, 184))
    if has_tab:
        _draw_corner_tab(draw, mask_draw)

    image_dir = output_dir / split
    mask_dir = output_dir / "masks"
    ensure_directory(image_dir)
    ensure_directory(mask_dir)
    image_path = image_dir / f"{sample_id}.png"
    mask_path = mask_dir / f"{sample_id}_tab_mask.png"
    image.save(image_path)
    mask.save(mask_path)
    return NuisanceBoardSample(
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        split=split,
        label=label,
        has_tab=has_tab,
        tab_region=BoundingBox(x=276, y=24, width=24, height=34),
        semantic_note=semantic_note,
    )


def generate_nuisance_board_dataset(
    output_dir: Path,
) -> tuple[list[NuisanceBoardSample], list[NuisanceBoardSample], list[NuisanceBoardSample]]:
    """Generate clean, contaminated, and query boards for the wrong-normal report."""

    clean_train = [
        _write_nuisance_sample(
            output_dir=output_dir,
            sample_id="clean_normal_000",
            split="train_clean",
            label="normal",
            has_tab=False,
            semantic_note="Clean nominal board without the acquisition tab.",
        ),
        _write_nuisance_sample(
            output_dir=output_dir,
            sample_id="clean_normal_001",
            split="train_clean",
            label="normal",
            has_tab=False,
            semantic_note="Second clean nominal board for provenance.",
        ),
    ]
    contaminated_train = [
        _write_nuisance_sample(
            output_dir=output_dir,
            sample_id="tabbed_normal_000",
            split="train_contaminated",
            label="normal",
            has_tab=True,
            semantic_note="Nominal board contaminated by a corner acquisition tab.",
        ),
        _write_nuisance_sample(
            output_dir=output_dir,
            sample_id="tabbed_normal_001",
            split="train_contaminated",
            label="normal",
            has_tab=True,
            semantic_note="Second contaminated nominal board for provenance.",
        ),
    ]
    query_samples = [
        _write_nuisance_sample(
            output_dir=output_dir,
            sample_id="query_clean_normal",
            split="test",
            label="normal",
            has_tab=False,
            semantic_note=(
                "A clean normal board. The contaminated bank can treat the missing "
                "corner tab as anomalous."
            ),
        ),
        _write_nuisance_sample(
            output_dir=output_dir,
            sample_id="query_tabbed_normal",
            split="test",
            label="normal",
            has_tab=True,
            semantic_note=(
                "A tabbed normal board. The clean bank flags the tab as acquisition "
                "nuisance."
            ),
        ),
    ]
    return clean_train, contaminated_train, query_samples
