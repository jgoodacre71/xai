"""Tiny deterministic image fixtures for tests and notebook smoke checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class SyntheticImageSample:
    """Metadata for a generated toy image."""

    sample_id: str
    image_path: Path
    label: str
    region: BoundingBox


def make_striped_fixture(output_dir: Path, sample_id: str = "striped-000") -> SyntheticImageSample:
    """Create a small RGB fixture with a bright central stripe.

    This fixture is intentionally simple. It provides deterministic pixels and a
    known region for tests before real datasets are available.
    """

    ensure_directory(output_dir)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, :] = (24, 32, 40)
    image[10:22, 14:18, :] = (220, 232, 96)

    image_path = output_dir / f"{sample_id}.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    return SyntheticImageSample(
        sample_id=sample_id,
        image_path=image_path,
        label="central_stripe",
        region=BoundingBox(x=14, y=10, width=4, height=12),
    )
