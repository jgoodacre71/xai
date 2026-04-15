"""Shared artefact contracts for demos and reports.

The contracts mirror the repository XAI contract: evidence, provenance,
counterfactual change, and stability. They deliberately stay model-agnostic so
classification, PatchCore, and robustness demos can share reporting code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned image region using pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Bounding-box origin must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding-box width and height must be positive.")

    @property
    def area(self) -> int:
        """Return the region area in pixels."""

        return self.width * self.height


@dataclass(frozen=True, slots=True)
class RegionScore:
    """Scored region used by evidence maps and anomaly localisation."""

    box: BoundingBox
    score: float
    label: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """A model output for one sample."""

    sample_id: str
    target: str
    predicted: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceArtefact:
    """Pixels, patches, regions, or features that materially drove an output."""

    sample_id: str
    method: str
    target: str
    heatmap_path: Path | None = None
    mask_path: Path | None = None
    top_regions: Sequence[RegionScore] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ProvenanceArtefact:
    """Training examples, prototypes, or nominal exemplars linked to a sample."""

    sample_id: str
    method: str
    reference_ids: Sequence[str]
    reference_scores: Sequence[float]
    reference_image_paths: Sequence[Path]
    reference_boxes: Sequence[BoundingBox] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        lengths = {
            "reference_ids": len(self.reference_ids),
            "reference_scores": len(self.reference_scores),
            "reference_image_paths": len(self.reference_image_paths),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Provenance reference lengths must match: {lengths}")
        if self.reference_boxes is not None and len(self.reference_boxes) != len(
            self.reference_ids
        ):
            raise ValueError("reference_boxes must match the number of reference ids.")


@dataclass(frozen=True, slots=True)
class CounterfactualArtefact:
    """A plausible intervention and its measured effect on the output."""

    sample_id: str
    method: str
    description: str
    before_score: float
    after_score: float
    output_path: Path | None = None

    @property
    def score_delta(self) -> float:
        """Return ``after_score - before_score`` for consistent reporting."""

        return self.after_score - self.before_score


@dataclass(frozen=True, slots=True)
class StabilityArtefact:
    """Explanation and prediction movement under perturbation or shift."""

    sample_id: str
    method: str
    perturbation_name: str
    prediction_shift: float
    explanation_shift: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Standard result object returned by a demo runner."""

    name: str
    summary: str
    predictions: Sequence[PredictionRecord] = field(default_factory=tuple)
    evidence: Sequence[EvidenceArtefact] = field(default_factory=tuple)
    provenance: Sequence[ProvenanceArtefact] = field(default_factory=tuple)
    counterfactuals: Sequence[CounterfactualArtefact] = field(default_factory=tuple)
    stability: Sequence[StabilityArtefact] = field(default_factory=tuple)
    metrics: Mapping[str, float] = field(default_factory=dict)
    assets: Mapping[str, Path] = field(default_factory=dict)


class DemoRunner(Protocol[ConfigT_contra]):
    """Protocol implemented by concrete demo runners."""

    def run(self, config: ConfigT_contra) -> DemoResult:
        """Run a demo from a typed configuration object."""
        ...
