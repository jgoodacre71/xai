from pathlib import Path

import pytest

from xai_demo_suite.explain import (
    BoundingBox,
    CounterfactualArtefact,
    ProvenanceArtefact,
)


def test_bounding_box_requires_positive_size() -> None:
    with pytest.raises(ValueError, match="width and height"):
        BoundingBox(x=0, y=0, width=0, height=4)


def test_bounding_box_area() -> None:
    assert BoundingBox(x=2, y=3, width=4, height=5).area == 20


def test_provenance_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        ProvenanceArtefact(
            sample_id="sample-1",
            method="nearest-normal-patch",
            reference_ids=["a", "b"],
            reference_scores=[0.1],
            reference_image_paths=[Path("a.png"), Path("b.png")],
        )


def test_counterfactual_score_delta() -> None:
    artefact = CounterfactualArtefact(
        sample_id="sample-1",
        method="mask-top-patch",
        description="Mask the highest scoring anomalous patch.",
        before_score=0.8,
        after_score=0.3,
    )

    assert artefact.score_delta == pytest.approx(-0.5)
