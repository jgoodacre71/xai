"""Static report slice for the MVTec AD bottle PatchCore demo."""

from __future__ import annotations

import html
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
    PatchFeatureExtractor,
    TorchvisionBackbonePatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchScore
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import (
    draw_box_on_image,
    save_mask_overlay,
    save_patch_crop,
    save_score_overlay,
)


@dataclass(frozen=True, slots=True)
class PatchCoreBottleReportConfig:
    """Configuration for the first static PatchCore bottle report."""

    manifest_path: Path = Path("data/processed/mvtec_ad/bottle/manifest.jsonl")
    output_dir: Path = Path("outputs/patchcore_bottle")
    cache_path: Path = Path("data/artefacts/patchcore/bottle/report_resnet18_bank.npz")
    max_train: int = 2
    test_index: int = 0
    max_examples: int = 3
    patch_size: int = 128
    stride: int = 128
    top_k: int = 3
    input_size: int = 64
    batch_size: int = 8
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
    mask_overlap: PatchMaskOverlap | None = None


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _build_default_extractor(config: PatchCoreBottleReportConfig) -> PatchFeatureExtractor:
    return TorchvisionBackbonePatchFeatureExtractor(
        input_size=config.input_size,
        batch_size=config.batch_size,
        weights_name=None,
    )


def _build_or_load_bank(
    config: PatchCoreBottleReportConfig,
    extractor: PatchFeatureExtractor,
) -> PatchCoreMemoryBank:
    records = load_image_manifest(config.manifest_path)
    train_records = filter_manifest_records(records, split="train", defect_type="good")[
        : config.max_train
    ]
    if not train_records:
        raise ValueError("No nominal training records found for MVTec AD bottle.")

    if config.use_cache and config.cache_path.exists():
        memory_bank = load_memory_bank(config.cache_path)
    else:
        memory_bank = build_patchcore_memory_bank(
            train_records,
            extractor=extractor,
            patch_size=config.patch_size,
            stride=config.stride,
        )
        save_memory_bank(memory_bank, config.cache_path)
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
        <figcaption>Top scored query patch. Distance: {score.distance:.6f}</figcaption>
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
"""


def _render_html(
    *,
    config: PatchCoreBottleReportConfig,
    examples: list[PatchCoreBottleExampleReport],
    output_path: Path,
) -> None:
    example_sections = "\n".join(
        _render_example_section(example=example, output_path=output_path)
        for example in examples
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Bottle Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
    }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
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
  </style>
</head>
<body>
<main>
  <h1>PatchCore Bottle Report</h1>
  <p>
    This report is generated from package code. It shows nearest-normal patch
    provenance for {len(examples)} selected MVTec AD bottle anomaly example(s).
  </p>

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>manifest: <code>{html.escape(config.manifest_path.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(config.cache_path.as_posix())}</code></li>
      <li>examples: {len(examples)} selected from test index {config.test_index}</li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>torchvision_resnet18</code>, random weights</li>
    </ul>
  </section>

{example_sections}
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _build_demo_card(output_path: Path, assets: dict[str, Path]) -> DemoCard:
    return DemoCard(
        title="Demo 03 - PatchCore on MVTec AD bottle",
        task="Unsupervised industrial anomaly detection on the MVTec AD bottle category.",
        model="PatchCore-style nearest-neighbour memory bank with ResNet-18 patch-crop features.",
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
        failure_mode=(
            "Current features use random-weight ResNet-18 patch crops and a coarse "
            "patch grid, so this is a pipeline/provenance slice rather than final "
            "anomaly-detection quality."
        ),
        intervention=(
            "Replace the top scored query patch with its nearest normal patch and "
            "recompute the local score as a didactic counterfactual probe."
        ),
        remaining_caveats=(
            "Not a calibrated severity model.",
            "Not a causal proof.",
            "No coreset selection or multi-scale feature-map PatchCore yet.",
            "Coarse overlay is not pixel-level anomaly-map interpolation.",
            "Top-patch mask overlap is a coarse verification check, not a full benchmark.",
        ),
        report_path=output_path,
        figure_paths=(
            assets["score_overlay"],
            assets["query_crop"],
            assets["normal_crop_1"],
            assets["counterfactual"],
        ),
    )


def build_patchcore_bottle_report(
    config: PatchCoreBottleReportConfig,
    extractor: PatchFeatureExtractor | None = None,
) -> Path:
    """Build a static HTML report for selected MVTec AD bottle examples."""

    ensure_directory(config.output_dir)
    extractor = extractor or _build_default_extractor(config)
    memory_bank = _build_or_load_bank(config, extractor)
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
                mask_overlap=mask_overlap,
            )
        )

    output_path = config.output_dir / "index.html"
    _render_html(
        config=config,
        examples=example_reports,
        output_path=output_path,
    )
    card = _build_demo_card(output_path=output_path, assets=example_reports[0].assets)
    save_demo_card(card, config.output_dir)
    save_demo_index((card,), config.output_dir.parent / "index.html")
    return output_path
