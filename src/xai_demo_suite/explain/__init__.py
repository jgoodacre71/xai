"""Explanation contracts and reusable explainability helpers."""

from xai_demo_suite.explain.contracts import (
    BoundingBox,
    CounterfactualArtefact,
    DemoResult,
    DemoRunner,
    EvidenceArtefact,
    PredictionRecord,
    ProvenanceArtefact,
    RegionScore,
    StabilityArtefact,
)
from xai_demo_suite.explain.counterfactuals import (
    make_patch_replacement_artefact,
    replace_patch_from_source,
)

__all__ = [
    "BoundingBox",
    "CounterfactualArtefact",
    "DemoResult",
    "DemoRunner",
    "EvidenceArtefact",
    "PredictionRecord",
    "ProvenanceArtefact",
    "RegionScore",
    "StabilityArtefact",
    "make_patch_replacement_artefact",
    "replace_patch_from_source",
]
