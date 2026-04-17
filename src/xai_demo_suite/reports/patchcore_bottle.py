"""Static report slice for the MVTec AD bottle PatchCore demo."""

from __future__ import annotations

import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.manifests import (
    ImageManifestRecord,
    filter_manifest_records,
    load_image_manifest,
)
from xai_demo_suite.evaluate.localisation import PatchMaskOverlap, measure_patch_mask_overlap
from xai_demo_suite.explain.contracts import CounterfactualArtefact
from xai_demo_suite.explain.counterfactuals import (
    make_patch_replacement_artefact,
    replace_patch_from_source,
)
from xai_demo_suite.models.patchcore import (
    ColourTexturePatchFeatureExtractor,
    MeanRGBPatchFeatureExtractor,
    PatchFeatureExtractor,
    TorchvisionBackbonePatchFeatureExtractor,
    TorchvisionFeatureMapPatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    reduce_memory_bank_coreset,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchScore
from xai_demo_suite.reports.build_metadata import BuildMetadata, make_build_metadata
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.reports.report_chrome import (
    ReportBrief,
    ReportLink,
    render_related_reports,
    render_report_brief,
    render_report_header,
    report_chrome_css,
)
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import (
    draw_box_on_image,
    save_mask_overlay,
    save_patch_crop,
    save_score_overlay,
)

DEFAULT_BOTTLE_CACHE_PATH = Path("data/artefacts/patchcore/bottle/report_colour_texture_bank.npz")


@dataclass(frozen=True, slots=True)
class PatchCoreBottleReportConfig:
    """Configuration for the first static PatchCore bottle report."""

    manifest_path: Path = Path("data/processed/mvtec_ad/bottle/manifest.jsonl")
    output_dir: Path = Path("outputs/patchcore_bottle")
    cache_path: Path = DEFAULT_BOTTLE_CACHE_PATH
    feature_extractor_name: str = "colour_texture"
    max_train: int = 10
    test_index: int = 0
    max_examples: int = 3
    patch_size: int = 128
    stride: int = 64
    top_k: int = 3
    input_size: int = 224
    batch_size: int = 8
    coreset_size: int | None = None
    coreset_seed: int = 0
    max_benchmark_records: int | None = None
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class PatchCoreBottleExampleReport:
    """Rendered report data for one selected anomalous bottle example."""

    example_number: int
    query_record: ImageManifestRecord
    score: PatchScore
    all_scores: list[PatchScore]
    assets: dict[str, Path]
    counterfactual: CounterfactualArtefact
    nominal_score_percentile: float | None
    mask_overlap: PatchMaskOverlap | None = None


@dataclass(frozen=True, slots=True)
class PatchCoreBottleBenchmarkRecord:
    """Dataset-level top-score diagnostic for one MVTec AD test image."""

    sample_id: str
    defect_type: str
    is_anomalous: bool
    top_score: float
    mask_overlap: PatchMaskOverlap | None = None


@dataclass(frozen=True, slots=True)
class PatchCoreBottleBenchmarkReport:
    """Local benchmark-style diagnostics for the report model path."""

    records: tuple[PatchCoreBottleBenchmarkRecord, ...]
    image_auc: float | None

    @property
    def good_count(self) -> int:
        """Return the number of scored nominal test images."""

        return sum(1 for record in self.records if not record.is_anomalous)

    @property
    def anomalous_count(self) -> int:
        """Return the number of scored anomalous test images."""

        return sum(1 for record in self.records if record.is_anomalous)


@dataclass(frozen=True, slots=True)
class PatchCoreBottleNominalControlReport:
    """Low-score nominal control example for live comparison."""

    query_record: ImageManifestRecord
    score: PatchScore
    all_scores: list[PatchScore]
    assets: dict[str, Path]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _build_default_extractor(config: PatchCoreBottleReportConfig) -> PatchFeatureExtractor:
    if config.feature_extractor_name == "colour_texture":
        return ColourTexturePatchFeatureExtractor()
    if config.feature_extractor_name == "mean_rgb":
        return MeanRGBPatchFeatureExtractor()
    if config.feature_extractor_name == "resnet18_random":
        return TorchvisionBackbonePatchFeatureExtractor(
            input_size=config.input_size,
            batch_size=config.batch_size,
            weights_name=None,
        )
    if config.feature_extractor_name == "feature_map_resnet18_random":
        return TorchvisionFeatureMapPatchFeatureExtractor(
            backbone_name="resnet18",
            feature_name="feature_map_resnet18_random_layer2_layer3",
            input_size=config.input_size,
            layer_names=("layer2", "layer3"),
            weights_name=None,
        )
    if config.feature_extractor_name == "feature_map_resnet18_pretrained":
        return TorchvisionFeatureMapPatchFeatureExtractor(
            backbone_name="resnet18",
            feature_name="feature_map_resnet18_imagenet_layer2_layer3",
            input_size=config.input_size,
            layer_names=("layer2", "layer3"),
            weights_name="DEFAULT",
        )
    if config.feature_extractor_name == "feature_map_wide_resnet50_2_random":
        return TorchvisionFeatureMapPatchFeatureExtractor(
            backbone_name="wide_resnet50_2",
            feature_name="feature_map_wide_resnet50_2_random_layer2_layer3",
            input_size=config.input_size,
            layer_names=("layer2", "layer3"),
            weights_name=None,
        )
    if config.feature_extractor_name == "feature_map_wide_resnet50_2_pretrained":
        return TorchvisionFeatureMapPatchFeatureExtractor(
            backbone_name="wide_resnet50_2",
            feature_name="feature_map_wide_resnet50_2_imagenet_layer2_layer3",
            input_size=config.input_size,
            layer_names=("layer2", "layer3"),
            weights_name="DEFAULT",
        )
    raise ValueError(
        "Unsupported feature_extractor_name. Expected one of: "
        "colour_texture, mean_rgb, resnet18_random, "
        "feature_map_resnet18_random, feature_map_resnet18_pretrained, "
        "feature_map_wide_resnet50_2_random, feature_map_wide_resnet50_2_pretrained."
    )


def _memory_bank_matches_config(
    *,
    memory_bank: PatchCoreMemoryBank,
    extractor: PatchFeatureExtractor,
    config: PatchCoreBottleReportConfig,
) -> bool:
    if memory_bank.feature_name != extractor.feature_name:
        return False
    return not (
        config.coreset_size is not None and len(memory_bank.metadata) > config.coreset_size
    )


def _safe_cache_component(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _resolve_cache_path(
    *,
    config: PatchCoreBottleReportConfig,
    extractor: PatchFeatureExtractor,
) -> Path:
    if config.cache_path != DEFAULT_BOTTLE_CACHE_PATH:
        return config.cache_path
    coreset_suffix = (
        f"_coreset_{config.coreset_size}" if config.coreset_size is not None else ""
    )
    return config.cache_path.with_name(
        f"report_{_safe_cache_component(extractor.feature_name)}{coreset_suffix}_bank.npz"
    )


def _build_or_load_bank(
    config: PatchCoreBottleReportConfig,
    extractor: PatchFeatureExtractor,
) -> PatchCoreMemoryBank:
    cache_path = _resolve_cache_path(config=config, extractor=extractor)
    records = load_image_manifest(config.manifest_path)
    train_records = filter_manifest_records(records, split="train", defect_type="good")[
        : config.max_train
    ]
    if not train_records:
        raise ValueError("No nominal training records found for MVTec AD bottle.")

    if config.use_cache and cache_path.exists():
        memory_bank = load_memory_bank(cache_path)
        if _memory_bank_matches_config(
            memory_bank=memory_bank,
            extractor=extractor,
            config=config,
        ):
            return memory_bank

    memory_bank = build_patchcore_memory_bank(
        train_records,
        extractor=extractor,
        patch_size=config.patch_size,
        stride=config.stride,
    )
    if config.coreset_size is not None:
        memory_bank = reduce_memory_bank_coreset(
            memory_bank,
            max_patches=config.coreset_size,
            seed=config.coreset_seed,
        )
    save_memory_bank(memory_bank, cache_path)
    return memory_bank


def _select_query_records(config: PatchCoreBottleReportConfig) -> list[ImageManifestRecord]:
    if config.max_examples < 1:
        raise ValueError("max_examples must be at least 1.")
    records = load_image_manifest(config.manifest_path)
    query_records = filter_manifest_records(records, split="test", is_anomalous=True)
    if not query_records:
        raise ValueError("No anomalous test records found for MVTec AD bottle.")
    selected = query_records[config.test_index : config.test_index + config.max_examples]
    if not selected:
        raise ValueError(
            f"test_index {config.test_index} is out of range for {len(query_records)} records."
        )
    return selected


def _select_benchmark_records(config: PatchCoreBottleReportConfig) -> list[ImageManifestRecord]:
    records = load_image_manifest(config.manifest_path)
    test_records = filter_manifest_records(records, split="test")
    if config.max_benchmark_records is not None:
        if config.max_benchmark_records <= 0:
            raise ValueError("max_benchmark_records must be positive when set.")
        return test_records[: config.max_benchmark_records]
    return test_records


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute binary ROC AUC with average ranks for tied scores."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    positive_count = sum(1 for label in labels if label)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for ranked_index in range(index, end):
            original_index = ranked[ranked_index][0]
            ranks[original_index] = average_rank
        index = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)) / (
        positive_count * negative_count
    )


def _build_benchmark_report(
    *,
    config: PatchCoreBottleReportConfig,
    memory_bank: PatchCoreMemoryBank,
    extractor: PatchFeatureExtractor,
) -> PatchCoreBottleBenchmarkReport | None:
    benchmark_records = _select_benchmark_records(config)
    if not benchmark_records:
        return None

    scored_records: list[PatchCoreBottleBenchmarkRecord] = []
    for record in benchmark_records:
        scores = score_image_with_extractor(
            sample_id=record.sample_id,
            image_path=record.image_path,
            memory_bank=memory_bank,
            extractor=extractor,
            patch_size=config.patch_size,
            stride=config.stride,
            top_k=config.top_k,
        )
        if not scores:
            continue
        top_score = scores[0]
        mask_overlap = None
        if record.mask_path is not None and record.mask_path.exists():
            mask_overlap = measure_patch_mask_overlap(
                mask_path=record.mask_path,
                patch_box=top_score.query_box,
                image_path=record.image_path,
            )
        scored_records.append(
            PatchCoreBottleBenchmarkRecord(
                sample_id=record.sample_id,
                defect_type=record.defect_type,
                is_anomalous=record.is_anomalous,
                top_score=top_score.distance,
                mask_overlap=mask_overlap,
            )
        )

    if not scored_records:
        return None
    return PatchCoreBottleBenchmarkReport(
        records=tuple(scored_records),
        image_auc=_roc_auc(
            labels=[record.is_anomalous for record in scored_records],
            scores=[record.top_score for record in scored_records],
        ),
    )


def _prefixed_asset_name(asset_prefix: str, name: str) -> str:
    return f"{asset_prefix}{name}" if asset_prefix else name


def _write_assets(
    *,
    score: PatchScore,
    all_scores: list[PatchScore],
    counterfactual_path: Path | None,
    mask_path: Path | None,
    output_dir: Path,
    asset_prefix: str,
) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    assets["score_overlay"] = save_score_overlay(
        image_path=score.image_path,
        scores=all_scores,
        output_path=_asset_path(
            output_dir,
            _prefixed_asset_name(asset_prefix, "score_overlay.png"),
        ),
    )
    assets["query_box"] = draw_box_on_image(
        image_path=score.image_path,
        box=score.query_box,
        output_path=_asset_path(
            output_dir,
            _prefixed_asset_name(asset_prefix, "query_box.png"),
        ),
    )
    assets["query_crop"] = save_patch_crop(
        image_path=score.image_path,
        box=score.query_box,
        output_path=_asset_path(
            output_dir,
            _prefixed_asset_name(asset_prefix, "query_patch.png"),
        ),
        scale=3,
    )
    for index, neighbour in enumerate(score.nearest, start=1):
        assets[f"normal_crop_{index}"] = save_patch_crop(
            image_path=neighbour.metadata.source_path,
            box=neighbour.metadata.box,
            output_path=_asset_path(
                output_dir,
                _prefixed_asset_name(asset_prefix, f"normal_patch_{index}.png"),
            ),
            scale=3,
        )
        assets[f"normal_box_{index}"] = draw_box_on_image(
            image_path=neighbour.metadata.source_path,
            box=neighbour.metadata.box,
            output_path=_asset_path(
                output_dir,
                _prefixed_asset_name(asset_prefix, f"normal_source_{index}.png"),
            ),
            colour=(30, 120, 220),
        )
    if counterfactual_path is not None:
        assets["counterfactual"] = draw_box_on_image(
            image_path=counterfactual_path,
            box=score.query_box,
            output_path=_asset_path(
                output_dir,
                _prefixed_asset_name(asset_prefix, "counterfactual_box.png"),
            ),
            colour=(40, 160, 80),
        )
    if mask_path is not None:
        assets["mask_overlay"] = save_mask_overlay(
            image_path=score.image_path,
            mask_path=mask_path,
            output_path=_asset_path(
                output_dir,
                _prefixed_asset_name(asset_prefix, "mask_overlay.png"),
            ),
        )
    return assets


def _build_counterfactual_preview(
    *,
    score: PatchScore,
    memory_bank: PatchCoreMemoryBank,
    extractor: PatchFeatureExtractor,
    config: PatchCoreBottleReportConfig,
    asset_prefix: str,
) -> CounterfactualArtefact:
    nearest = score.nearest[0]
    output_path = _asset_path(
        config.output_dir,
        _prefixed_asset_name(asset_prefix, "counterfactual_replacement.png"),
    )
    replace_patch_from_source(
        image_path=score.image_path,
        query_box=score.query_box,
        source_image_path=nearest.metadata.source_path,
        source_box=nearest.metadata.box,
        output_path=output_path,
    )
    rescored = score_image_with_extractor(
        sample_id=f"{score.sample_id}/counterfactual",
        image_path=output_path,
        memory_bank=memory_bank,
        extractor=extractor,
        patch_size=config.patch_size,
        stride=config.stride,
        top_k=config.top_k,
    )
    matching_scores = [
        item
        for item in rescored
        if item.query_box.x == score.query_box.x and item.query_box.y == score.query_box.y
    ]
    after_score = matching_scores[0].distance if matching_scores else rescored[0].distance
    return make_patch_replacement_artefact(
        sample_id=score.sample_id,
        before_score=score.distance,
        after_score=after_score,
        output_path=output_path,
        description=(
            "Replace the top scored query patch with its nearest normal patch "
            "from the current memory bank."
        ),
    )


def _nominal_score_percentile(score: float, nominal_scores: list[float]) -> float | None:
    """Return the percentile position against nominal top scores."""

    if not nominal_scores:
        return None
    count_at_or_below = sum(1 for nominal_score in nominal_scores if nominal_score <= score)
    return count_at_or_below / len(nominal_scores)


def _render_example_section(
    *,
    example: PatchCoreBottleExampleReport,
    output_path: Path,
) -> str:
    score = example.score
    all_scores = example.all_scores
    assets = example.assets
    counterfactual = example.counterfactual
    rows: list[str] = []
    for rank, neighbour in enumerate(score.nearest, start=1):
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(neighbour.metadata.source_image_id)}</td>"
            f"<td>{neighbour.distance:.6f}</td>"
            f"<td>{html.escape(str(neighbour.metadata.box))}</td>"
            f"<td>{html.escape(neighbour.metadata.source_path.as_posix())}</td>"
            "</tr>"
        )

    top_scores = "\n".join(
        f"<li>patch {index + 1}: distance {patch_score.distance:.6f}, "
        f"box {html.escape(str(patch_score.query_box))}</li>"
        for index, patch_score in enumerate(all_scores[:5])
    )

    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    query_box_src = rel(assets["query_box"])
    query_crop_src = rel(assets["query_crop"])
    score_overlay_src = rel(assets["score_overlay"])
    counterfactual_src = rel(assets["counterfactual"])
    neighbour_blocks: list[str] = []
    for index in range(1, len(score.nearest) + 1):
        crop_src = rel(assets[f"normal_crop_{index}"])
        box_src = rel(assets[f"normal_box_{index}"])
        distance = score.nearest[index - 1].distance
        neighbour_blocks.append(
            f"""
      <figure>
        <img src="{crop_src}" alt="Nearest normal patch {index}">
        <figcaption>
          Nearest normal patch {index}; distance {distance:.6f}
        </figcaption>
      </figure>
      <figure>
        <img src="{box_src}" alt="Source image for nearest normal patch {index}">
        <figcaption>Full source image with patch box {index}.</figcaption>
      </figure>
      """
        )
    neighbour_figures = "\n".join(neighbour_blocks)
    sample_id = html.escape(example.query_record.sample_id)
    image_path = html.escape(example.query_record.image_path.as_posix())
    defect_type = html.escape(example.query_record.defect_type)
    mask_check = _render_mask_check(example=example, output_path=output_path)
    percentile_text = "Nominal reference percentile not available"
    if example.nominal_score_percentile is not None:
        percentile_text = (
            f"{100.0 * example.nominal_score_percentile:.1f}th percentile of nominal "
            "top-patch scores"
        )
    counterfactual_drop = 0.0
    if counterfactual.before_score != 0.0:
        counterfactual_drop = abs(counterfactual.score_delta) / counterfactual.before_score

    return f"""
  <section class="example">
    <h2>Example {example.example_number}: {defect_type}</h2>
    <p>
      Sample <code>{sample_id}</code> from
      <code>{image_path}</code>.
    </p>

    <h3>Top Scored Patch</h3>
    <div class="grid">
      <figure>
        <img src="{score_overlay_src}" alt="Coarse anomaly-map overlay">
        <figcaption>
          Coarse patch-score anomaly map. Brighter red means higher patch
          distance to the nominal memory bank.
        </figcaption>
      </figure>
      <figure>
        <img src="{query_box_src}" alt="Input image with top scored patch box">
        <figcaption>Input image with the top scored patch highlighted.</figcaption>
      </figure>
      <figure>
        <img src="{query_crop_src}" alt="Top scored query patch crop">
        <figcaption>
          Top scored query patch. Distance: {score.distance:.6f} |
          {html.escape(percentile_text)}
        </figcaption>
      </figure>
    </div>

{mask_check}

    <h3>Nearest Normal Patch Evidence</h3>
    <div class="grid">
      {neighbour_figures}
    </div>

    <h3>Counterfactual Patch Replacement</h3>
    <div class="grid">
      <figure>
        <img src="{counterfactual_src}" alt="Counterfactual replacement preview">
        <figcaption>
          Top query patch replaced with the nearest normal source patch. This is
          a didactic probe, not causal proof.
        </figcaption>
      </figure>
    </div>
    <ul>
      <li>Before score: {counterfactual.before_score:.6f}</li>
      <li>After score: {counterfactual.after_score:.6f}</li>
      <li>Delta: {counterfactual.score_delta:.6f}</li>
      <li>Score reduction: {100.0 * counterfactual_drop:.1f}%</li>
    </ul>

    <h3>Distance Summary</h3>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Source image id</th>
          <th>Distance</th>
          <th>Source box</th>
          <th>Source path</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>

    <h3>Top Query Patch Distances</h3>
    <ol>{top_scores}</ol>
  </section>
"""


def _format_percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _render_mask_check(
    *,
    example: PatchCoreBottleExampleReport,
    output_path: Path,
) -> str:
    if example.mask_overlap is None:
        return ""

    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    overlap = example.mask_overlap
    mask_overlay_src = rel(example.assets["mask_overlay"])
    mask_path = html.escape(overlap.mask_path.as_posix())
    intersects = "yes" if overlap.intersects_mask else "no"
    patch_fraction = _format_percentage(overlap.patch_mask_fraction)
    mask_fraction = _format_percentage(overlap.mask_covered_fraction)
    patch_overlap_text = f"{overlap.intersection_area} / {overlap.patch_area} ({patch_fraction})"
    mask_coverage_text = f"{overlap.intersection_area} / {overlap.mask_area} ({mask_fraction})"
    return f"""
    <h3>Ground Truth Localisation Check</h3>
    <div class="grid">
      <figure>
        <img src="{mask_overlay_src}" alt="Ground-truth anomaly mask overlay">
        <figcaption>
          Ground-truth anomaly mask overlay from <code>{mask_path}</code>.
        </figcaption>
      </figure>
    </div>
    <ul>
      <li>Top patch intersects mask: {intersects}</li>
      <li>Patch pixels overlapping mask: {patch_overlap_text}</li>
      <li>Mask covered by top patch: {mask_coverage_text}</li>
    </ul>
    <p>
      The top patch is a witness patch, not a full defect segmentation. Strong overlap with the
      anomalous region can still cover only part of the total defect extent.
    </p>
"""


def _render_overview(examples: list[PatchCoreBottleExampleReport]) -> str:
    mask_examples = [example for example in examples if example.mask_overlap is not None]
    mask_hits = sum(
        1
        for example in mask_examples
        if example.mask_overlap is not None and example.mask_overlap.intersects_mask
    )
    mean_mask_coverage = (
        sum(
            example.mask_overlap.mask_covered_fraction
            for example in mask_examples
            if example.mask_overlap is not None
        )
        / len(mask_examples)
        if mask_examples
        else None
    )
    mean_counterfactual_delta = sum(
        example.counterfactual.score_delta for example in examples
    ) / len(examples)
    mean_counterfactual_drop = sum(
        abs(example.counterfactual.score_delta) / example.counterfactual.before_score
        if example.counterfactual.before_score != 0.0
        else 0.0
        for example in examples
    ) / len(examples)

    rows: list[str] = []
    for example in examples:
        overlap = example.mask_overlap
        if overlap is None:
            intersects = "not available"
            patch_overlap = "not available"
            mask_coverage = "not available"
        else:
            intersects = "yes" if overlap.intersects_mask else "no"
            patch_overlap = _format_percentage(overlap.patch_mask_fraction)
            mask_coverage = _format_percentage(overlap.mask_covered_fraction)
        rows.append(
            "<tr>"
            f"<td>{example.example_number}</td>"
            f"<td>{html.escape(example.query_record.defect_type)}</td>"
            f"<td>{example.score.distance:.6f}</td>"
            f"<td>{intersects}</td>"
            f"<td>{patch_overlap}</td>"
            f"<td>{mask_coverage}</td>"
            f"<td>{example.counterfactual.score_delta:.6f}</td>"
            "</tr>"
        )

    mean_mask_text = (
        _format_percentage(mean_mask_coverage)
        if mean_mask_coverage is not None
        else "not available"
    )
    return f"""
  <section>
    <h2>Selected Example Overview</h2>
    <ul>
      <li>Mask-intersection hits: {mask_hits} / {len(mask_examples)} masked examples</li>
      <li>Mean mask covered by top patch: {mean_mask_text}</li>
      <li>Mean counterfactual score delta: {mean_counterfactual_delta:.6f}</li>
      <li>Mean counterfactual score reduction: {100.0 * mean_counterfactual_drop:.1f}%</li>
    </ul>
    <table>
      <thead>
        <tr>
          <th>Example</th>
          <th>Defect type</th>
          <th>Top patch score</th>
          <th>Intersects mask</th>
          <th>Patch overlap</th>
          <th>Mask covered</th>
          <th>Counterfactual delta</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p>
      These are selected-example diagnostics only, not benchmark metrics.
    </p>
  </section>
"""


def _render_nominal_control_section(
    *,
    control: PatchCoreBottleNominalControlReport | None,
    output_path: Path,
) -> str:
    if control is None:
        return ""

    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    return f"""
  <section>
    <h2>Nominal Control Example</h2>
    <p>
      This low-score good example anchors the scale: the same patch pipeline on a nominal bottle
      produces a much lower top-patch score and no obvious local witness patch.
    </p>
    <div class="grid">
      <figure>
        <img src="{rel(control.assets["score_overlay"])}" alt="Nominal control score overlay">
        <figcaption>Nominal control anomaly map.</figcaption>
      </figure>
      <figure>
        <img src="{rel(control.assets["query_box"])}" alt="Nominal control top patch box">
        <figcaption>Top patch on the nominal control image.</figcaption>
      </figure>
      <figure>
        <img src="{rel(control.assets["query_crop"])}" alt="Nominal control top patch crop">
        <figcaption>Top nominal patch. Distance: {control.score.distance:.6f}</figcaption>
      </figure>
    </div>
  </section>
"""


def _render_benchmark(benchmark: PatchCoreBottleBenchmarkReport | None) -> str:
    if benchmark is None:
        return """
  <section>
    <h2>Test-Split Benchmark Diagnostics</h2>
    <p>No test-split benchmark records were available for this run.</p>
  </section>
"""

    auc_text = f"{benchmark.image_auc:.3f}" if benchmark.image_auc is not None else "not available"
    mask_records = [
        record for record in benchmark.records if record.mask_overlap is not None
    ]
    mask_hits = sum(
        1
        for record in mask_records
        if record.mask_overlap is not None and record.mask_overlap.intersects_mask
    )
    mean_mask_coverage = (
        sum(
            record.mask_overlap.mask_covered_fraction
            for record in mask_records
            if record.mask_overlap is not None
        )
        / len(mask_records)
        if mask_records
        else None
    )
    mean_mask_text = (
        _format_percentage(mean_mask_coverage)
        if mean_mask_coverage is not None
        else "not available"
    )

    by_defect: dict[str, list[PatchCoreBottleBenchmarkRecord]] = defaultdict(list)
    for record in benchmark.records:
        by_defect[record.defect_type].append(record)

    rows: list[str] = []
    for defect_type, records in sorted(by_defect.items()):
        scores = [record.top_score for record in records]
        mask_subset = [record for record in records if record.mask_overlap is not None]
        defect_mask_hits = sum(
            1
            for record in mask_subset
            if record.mask_overlap is not None and record.mask_overlap.intersects_mask
        )
        mask_hit_text = (
            f"{defect_mask_hits} / {len(mask_subset)}" if mask_subset else "not available"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(defect_type)}</td>"
            f"<td>{len(records)}</td>"
            f"<td>{sum(1 for record in records if record.is_anomalous)}</td>"
            f"<td>{sum(scores) / len(scores):.6f}</td>"
            f"<td>{max(scores):.6f}</td>"
            f"<td>{mask_hit_text}</td>"
            "</tr>"
        )

    return f"""
  <section>
    <h2>Test-Split Benchmark Diagnostics</h2>
    <ul>
      <li>Scored test images: {len(benchmark.records)}</li>
      <li>Nominal / anomalous: {benchmark.good_count} / {benchmark.anomalous_count}</li>
      <li>Image-level ROC AUC from max patch score: {auc_text}</li>
      <li>Top patch intersects anomaly mask: {mask_hits} / {len(mask_records)} masked anomalies</li>
      <li>Mean mask covered by top patch: {mean_mask_text}</li>
    </ul>
    <table>
      <thead>
        <tr>
          <th>Defect type</th>
          <th>Images</th>
          <th>Anomalous</th>
          <th>Mean top score</th>
          <th>Max top score</th>
          <th>Mask hits</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p>
      These are local report diagnostics from the current patch grid and memory
      bank, not official PatchCore benchmark numbers or pixel-level AUROC.
    </p>
    <p>
      The top-patch overlap check is intentionally local. It tells you whether PatchCore found a
      useful witness patch, not whether it recovered the full defect extent.
    </p>
  </section>
"""


def _feature_path_description(feature_name: str) -> str:
    if "wide_resnet50_2_imagenet" in feature_name:
        return (
            "ImageNet-pretrained Torchvision WideResNet50-2 feature maps sampled from "
            "layer2 and layer3 at each patch centre."
        )
    if feature_name.startswith("feature_map_wide_resnet50_2_random"):
        return (
            "Torchvision WideResNet50-2 feature maps sampled from layer2 and layer3, "
            "using random weights for dependency/runtime testing."
        )
    if "imagenet" in feature_name:
        return (
            "ImageNet-pretrained Torchvision ResNet-18 feature maps sampled from "
            "layer2 and layer3 at each patch centre."
        )
    if feature_name.startswith("feature_map_resnet18_random"):
        return (
            "Torchvision ResNet-18 feature maps sampled from layer2 and layer3, "
            "using random weights for dependency/runtime testing."
        )
    if feature_name.startswith("torchvision_resnet18"):
        return "Torchvision ResNet-18 patch-crop features."
    if feature_name == "colour_texture":
        return "Deterministic colour, intensity, and edge-statistic patch features."
    if feature_name == "mean_rgb":
        return "Deterministic mean-RGB patch features."
    return feature_name


def _report_category_label(config: PatchCoreBottleReportConfig) -> str:
    return config.manifest_path.parent.name.replace("_", " ")


def _render_html(
    *,
    config: PatchCoreBottleReportConfig,
    examples: list[PatchCoreBottleExampleReport],
    benchmark: PatchCoreBottleBenchmarkReport | None,
    nominal_control: PatchCoreBottleNominalControlReport | None,
    feature_name: str,
    memory_bank_size: int,
    cache_path: Path,
    output_path: Path,
    build_metadata: BuildMetadata,
) -> None:
    category_label = _report_category_label(config)
    title_text = f"PatchCore on MVTec AD {category_label}"
    lede_text = (
        "Nearest-normal patch provenance, local counterfactual probes, and dataset-level "
        f"diagnostics for the MVTec AD {category_label} anomaly demo."
    )
    example_sections = "\n".join(
        _render_example_section(example=example, output_path=output_path)
        for example in examples
    )
    overview_section = _render_overview(examples)
    benchmark_section = _render_benchmark(benchmark)
    nominal_control_section = _render_nominal_control_section(
        control=nominal_control,
        output_path=output_path,
    )
    coreset_text = (
        f"{config.coreset_size} requested; {memory_bank_size} retained"
        if config.coreset_size is not None
        else f"not used; {memory_bank_size} patches retained"
    )
    feature_description = html.escape(_feature_path_description(feature_name))
    brief = ReportBrief(
        claim=(
            "PatchCore becomes much easier to trust and discuss once the anomaly score is tied "
            "to nearest-normal patch provenance and a simple local replacement probe."
        ),
        evidence=(
            "The strongest sequence is the selected example view plus the benchmark panel: the "
            "report shows the top anomalous patch, the nearest nominal source patch, and the "
            "effect of replacing the anomalous region."
        ),
        live_demo=(
            "Start with one bottle example, show the top patch and nearest-normal exemplar, then "
            "use the benchmark panel to show that the same run separates the full test split."
        ),
        boundary=(
            "This report is a strong local PatchCore-style implementation, not a full official "
            "benchmark reproduction or a calibrated severity model."
        ),
        related=(
            ReportLink(
                slug="patchcore_wrong_normal",
                title="Demo 04 - PatchCore Learns the Wrong Normal",
                reason=(
                    "Tests how the same provenance machinery behaves when normality itself is "
                    "contaminated."
                ),
            ),
            ReportLink(
                slug="patchcore_logic",
                title="Demo 07 - PatchCore Logical Anomaly Limits",
                reason="Shows where local novelty is not enough for rule-level inspection.",
            ),
            ReportLink(
                slug="explanation_drift",
                title="Demo 08 - Explanation Drift Under Shift",
                reason="Shows how anomaly evidence moves under nuisance perturbations.",
            ),
        ),
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title_text)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
      background: #f7f8fb;
    }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; background: #fff; padding: 20px; border: 1px solid #d8dee4; }}
    .example {{ border-top: 2px solid #d8dee4; padding-top: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    figure {{ margin: 0; border: 1px solid #d8dee4; padding: 10px; background: #fff; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ font-size: 13px; color: #52606d; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{
      border-bottom: 1px solid #d8dee4;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
    {report_chrome_css()}
  </style>
</head>
<body>
<main>
  {render_report_header(
      output_path=output_path,
      eyebrow="Demo 03 · PatchCore hero demo",
      title=title_text,
      lede=lede_text,
      build_metadata=build_metadata,
  )}
  {render_report_brief(brief)}

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>manifest: <code>{html.escape(config.manifest_path.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(cache_path.as_posix())}</code></li>
      <li>examples: {len(examples)} selected from test index {config.test_index}</li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>{html.escape(feature_name)}</code></li>
      <li>feature path: {feature_description}</li>
      <li>feature input size: {config.input_size}</li>
      <li>memory-bank coreset: {coreset_text}</li>
    </ul>
  </section>

{overview_section}

{benchmark_section}

{nominal_control_section}

{example_sections}
  {render_related_reports(output_path=output_path, heading="Where to go next", links=brief.related)}
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _build_demo_card(
    output_path: Path,
    assets: dict[str, Path],
    feature_name: str,
    coreset_size: int | None,
    category_label: str,
    build_metadata: BuildMetadata,
) -> DemoCard:
    uses_feature_map = feature_name.startswith("feature_map_resnet18") or feature_name.startswith(
        "feature_map_wide_resnet50_2"
    )
    uses_pretrained = "imagenet" in feature_name
    failure_mode = (
        "This is a practical local PatchCore implementation rather than an official "
        "benchmark reproduction; the anomaly map is still rendered on a coarse patch grid."
        if uses_pretrained
        else (
            "This run does not use pretrained feature-map weights, so it is useful for "
            "testing the provenance/reporting path but should not be treated as the strongest "
            "industrial anomaly detector."
            if uses_feature_map
            else (
                "Current default features are deterministic colour/texture statistics "
                "on a coarse patch grid. Use feature_map_resnet18_pretrained or "
                "feature_map_wide_resnet50_2_pretrained for the stronger deep-feature path."
            )
        )
    )
    coreset_caveat = (
        f"Nearest-normal exemplars are retrieved from the {coreset_size}-patch coreset."
        if coreset_size is not None
        else "Nearest-normal exemplars are retrieved from the retained local memory bank."
    )
    return DemoCard(
        title=f"Demo 03 - PatchCore on MVTec AD {category_label}",
        task=(
            f"Unsupervised industrial anomaly detection on the MVTec AD "
            f"{category_label} category."
        ),
        model=(
            "PatchCore-style nearest-neighbour memory bank with "
            f"{feature_name} patch features."
        ),
        explanation_methods=(
            "Coarse patch-score anomaly map",
            "Top anomalous patch crop",
            "Ground-truth mask overlap check",
            "Nearest-normal patch provenance",
            "Nearest-normal patch replacement probe",
        ),
        key_lesson=(
            "A PatchCore-style detector can make an anomaly score inspectable by "
            "showing which nominal patches were nearest to the suspicious region."
        ),
        failure_mode=failure_mode,
        intervention=(
            "Replace the top scored query patch with its nearest normal patch and "
            "recompute the local score as a didactic counterfactual probe."
        ),
        remaining_caveats=(
            "Not a calibrated severity model.",
            "Not a causal proof.",
            coreset_caveat,
            "Coarse overlay is not pixel-level anomaly-map interpolation.",
            (
                "Benchmark diagnostics use image-level max patch scores and "
                "top-patch overlap, not official pixel-level PatchCore metrics."
            ),
        ),
        report_path=output_path,
        figure_paths=(
            assets["score_overlay"],
            assets["query_crop"],
            assets["normal_crop_1"],
            assets["counterfactual"],
        ),
        build_metadata=build_metadata,
    )


def build_patchcore_bottle_report(
    config: PatchCoreBottleReportConfig,
    extractor: PatchFeatureExtractor | None = None,
) -> Path:
    """Build a static HTML report for selected MVTec AD bottle examples."""

    ensure_directory(config.output_dir)
    category_label = _report_category_label(config)
    extractor = extractor or _build_default_extractor(config)
    memory_bank = _build_or_load_bank(config, extractor)
    benchmark = _build_benchmark_report(
        config=config,
        memory_bank=memory_bank,
        extractor=extractor,
    )
    nominal_top_scores = [
        record.top_score for record in benchmark.records if not record.is_anomalous
    ] if benchmark is not None else []
    query_records = _select_query_records(config)
    use_asset_prefixes = len(query_records) > 1
    example_reports: list[PatchCoreBottleExampleReport] = []
    for example_number, query_record in enumerate(query_records, start=1):
        scores = score_image_with_extractor(
            sample_id=query_record.sample_id,
            image_path=query_record.image_path,
            memory_bank=memory_bank,
            extractor=extractor,
            patch_size=config.patch_size,
            stride=config.stride,
            top_k=config.top_k,
        )
        if not scores:
            raise ValueError(f"No query patch scores were produced for {query_record.sample_id}.")

        asset_prefix = f"example_{example_number}_" if use_asset_prefixes else ""
        top_score = scores[0]
        mask_overlap = None
        if query_record.mask_path is not None and query_record.mask_path.exists():
            mask_overlap = measure_patch_mask_overlap(
                mask_path=query_record.mask_path,
                patch_box=top_score.query_box,
                image_path=query_record.image_path,
            )
        counterfactual = _build_counterfactual_preview(
            score=top_score,
            memory_bank=memory_bank,
            extractor=extractor,
            config=config,
            asset_prefix=asset_prefix,
        )
        assets = _write_assets(
            score=top_score,
            all_scores=scores,
            counterfactual_path=counterfactual.output_path,
            mask_path=query_record.mask_path if mask_overlap is not None else None,
            output_dir=config.output_dir,
            asset_prefix=asset_prefix,
        )
        example_reports.append(
            PatchCoreBottleExampleReport(
                example_number=example_number,
                query_record=query_record,
                score=top_score,
                all_scores=scores,
                assets=assets,
                counterfactual=counterfactual,
                nominal_score_percentile=_nominal_score_percentile(
                    top_score.distance,
                    nominal_top_scores,
                ),
                mask_overlap=mask_overlap,
            )
        )
    nominal_control = None
    if benchmark is not None:
        good_records = [
            record
            for record in _select_benchmark_records(config)
            if not record.is_anomalous
        ]
        scored_good_examples: list[tuple[ImageManifestRecord, list[PatchScore]]] = []
        for record in good_records:
            scores = score_image_with_extractor(
                sample_id=record.sample_id,
                image_path=record.image_path,
                memory_bank=memory_bank,
                extractor=extractor,
                patch_size=config.patch_size,
                stride=config.stride,
                top_k=config.top_k,
            )
            if scores:
                scored_good_examples.append((record, scores))
        if scored_good_examples:
            control_record, control_scores = min(
                scored_good_examples,
                key=lambda item: item[1][0].distance,
            )
            nominal_control = PatchCoreBottleNominalControlReport(
                query_record=control_record,
                score=control_scores[0],
                all_scores=control_scores,
                assets=_write_assets(
                    score=control_scores[0],
                    all_scores=control_scores,
                    counterfactual_path=None,
                    mask_path=None,
                    output_dir=config.output_dir,
                    asset_prefix="nominal_control_",
                ),
            )

    output_path = config.output_dir / "index.html"
    build_metadata = make_build_metadata(
        data_mode="real",
        manifest_path=config.manifest_path,
        cache_enabled=config.use_cache,
    )
    _render_html(
        config=config,
        examples=example_reports,
        benchmark=benchmark,
        nominal_control=nominal_control,
        feature_name=extractor.feature_name,
        memory_bank_size=len(memory_bank.metadata),
        cache_path=_resolve_cache_path(config=config, extractor=extractor),
        output_path=output_path,
        build_metadata=build_metadata,
    )
    card = _build_demo_card(
        output_path=output_path,
        assets=example_reports[0].assets,
        feature_name=extractor.feature_name,
        coreset_size=config.coreset_size,
        category_label=category_label,
        build_metadata=build_metadata,
    )
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
