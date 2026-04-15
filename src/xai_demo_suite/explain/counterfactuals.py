"""Counterfactual probes for image explanations."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from xai_demo_suite.explain.contracts import BoundingBox, CounterfactualArtefact
from xai_demo_suite.utils.io import ensure_directory


def replace_patch_from_source(
    *,
    image_path: Path,
    query_box: BoundingBox,
    source_image_path: Path,
    source_box: BoundingBox,
    output_path: Path,
) -> Path:
    """Replace a query patch with a patch cropped from a recorded source image."""

    ensure_directory(output_path.parent)
    with Image.open(image_path) as query_image, Image.open(source_image_path) as source_image:
        output_image = query_image.convert("RGB")
        source_patch = source_image.convert("RGB").crop(
            (
                source_box.x,
                source_box.y,
                source_box.x + source_box.width,
                source_box.y + source_box.height,
            )
        )
        if source_patch.size != (query_box.width, query_box.height):
            source_patch = source_patch.resize(
                (query_box.width, query_box.height),
                Image.Resampling.BILINEAR,
            )
        output_image.paste(source_patch, (query_box.x, query_box.y))
        output_image.save(output_path)
    return output_path


def make_patch_replacement_artefact(
    *,
    sample_id: str,
    before_score: float,
    after_score: float,
    output_path: Path,
    description: str,
) -> CounterfactualArtefact:
    """Create a standard artefact for a patch replacement probe."""

    return CounterfactualArtefact(
        sample_id=sample_id,
        method="nearest-normal-patch-replacement",
        description=description,
        before_score=before_score,
        after_score=after_score,
        output_path=output_path,
    )
