"""Shared helpers for synthetic PatchCore limitation reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.data.synthetic import SlotBoardSample
from xai_demo_suite.evaluate.localisation import PatchMaskOverlap, measure_patch_mask_overlap
from xai_demo_suite.models.patchcore import (
    ColourTexturePatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchScore
from xai_demo_suite.vis.image_panels import (
    draw_box_on_image,
    save_mask_overlay,
    save_patch_crop,
    save_score_overlay,
)


@dataclass(frozen=True, slots=True)
class SyntheticPatchCoreExample:
    """Report data for one synthetic PatchCore limitation case."""

    sample: SlotBoardSample
    score: PatchScore
    all_scores: list[PatchScore]
    mask_overlap: PatchMaskOverlap
    assets: dict[str, Path]


def relative_path(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` for report HTML."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def format_percentage(value: float) -> str:
    """Format a ratio as a one-decimal percentage."""

    return f"{100.0 * value:.1f}%"


def sample_to_record(sample: SlotBoardSample, *, category: str) -> ImageManifestRecord:
    """Convert a synthetic slot-board sample into a manifest record."""

    return ImageManifestRecord(
        dataset="synthetic",
        category=category,
        split=sample.split,
        defect_type=sample.label,
        is_anomalous=sample.split != "train",
        image_path=sample.image_path,
        mask_path=sample.mask_path,
    )


def build_or_load_synthetic_bank(
    *,
    train_samples: list[SlotBoardSample],
    extractor: ColourTexturePatchFeatureExtractor,
    cache_path: Path,
    use_cache: bool,
    patch_size: int,
    stride: int,
    category: str,
) -> PatchCoreMemoryBank:
    """Build or load a synthetic PatchCore-style memory bank."""

    if use_cache and cache_path.exists():
        memory_bank = load_memory_bank(cache_path)
        if memory_bank.feature_name == extractor.feature_name:
            return memory_bank

    memory_bank = build_patchcore_memory_bank(
        [sample_to_record(sample, category=category) for sample in train_samples],
        extractor=extractor,
        patch_size=patch_size,
        stride=stride,
    )
    save_memory_bank(memory_bank, cache_path)
    return memory_bank


def write_synthetic_patchcore_assets(
    *,
    sample: SlotBoardSample,
    score: PatchScore,
    scores: list[PatchScore],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """Write common visual assets for one synthetic PatchCore example."""

    asset_dir = output_dir / "assets"
    assets: dict[str, Path] = {}
    assets["score_overlay"] = save_score_overlay(
        image_path=sample.image_path,
        scores=scores,
        output_path=asset_dir / f"{prefix}_score_overlay.png",
    )
    assets["query_box"] = draw_box_on_image(
        image_path=sample.image_path,
        box=score.query_box,
        output_path=asset_dir / f"{prefix}_query_box.png",
    )
    assets["query_crop"] = save_patch_crop(
        image_path=sample.image_path,
        box=score.query_box,
        output_path=asset_dir / f"{prefix}_query_patch.png",
        scale=2,
    )
    assets["mask_overlay"] = save_mask_overlay(
        image_path=sample.image_path,
        mask_path=sample.mask_path,
        output_path=asset_dir / f"{prefix}_mask_overlay.png",
    )
    nearest = score.nearest[0]
    assets["normal_crop"] = save_patch_crop(
        image_path=nearest.metadata.source_path,
        box=nearest.metadata.box,
        output_path=asset_dir / f"{prefix}_nearest_normal_patch.png",
        scale=2,
    )
    assets["normal_source"] = draw_box_on_image(
        image_path=nearest.metadata.source_path,
        box=nearest.metadata.box,
        output_path=asset_dir / f"{prefix}_nearest_normal_source.png",
        colour=(30, 120, 220),
    )
    return assets


def build_synthetic_patchcore_examples(
    *,
    eval_samples: list[SlotBoardSample],
    memory_bank: PatchCoreMemoryBank,
    extractor: ColourTexturePatchFeatureExtractor,
    output_dir: Path,
    patch_size: int,
    stride: int,
    top_k: int,
) -> list[SyntheticPatchCoreExample]:
    """Score synthetic examples and write common visual assets."""

    examples: list[SyntheticPatchCoreExample] = []
    for example_number, sample in enumerate(eval_samples, start=1):
        scores = score_image_with_extractor(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            memory_bank=memory_bank,
            extractor=extractor,
            patch_size=patch_size,
            stride=stride,
            top_k=top_k,
        )
        if not scores:
            raise ValueError(f"No scores produced for {sample.sample_id}.")
        top_score = scores[0]
        overlap = measure_patch_mask_overlap(
            mask_path=sample.mask_path,
            patch_box=top_score.query_box,
            image_path=sample.image_path,
        )
        assets = write_synthetic_patchcore_assets(
            sample=sample,
            score=top_score,
            scores=scores,
            output_dir=output_dir,
            prefix=f"example_{example_number}_{sample.sample_id}",
        )
        examples.append(
            SyntheticPatchCoreExample(
                sample=sample,
                score=top_score,
                all_scores=scores,
                mask_overlap=overlap,
                assets=assets,
            )
        )
    return examples
