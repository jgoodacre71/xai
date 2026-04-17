"""Static PatchCore limits report using deterministic synthetic slot boards."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.data.synthetic import SlotBoardSample, generate_slot_board_dataset
from xai_demo_suite.evaluate.localisation import PatchMaskOverlap, measure_patch_mask_overlap
from xai_demo_suite.models.patchcore import (
    ColourTexturePatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchScore
from xai_demo_suite.reports.build_metadata import (
    BuildMetadata,
    make_build_metadata,
    render_build_metadata_section,
)
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import (
    draw_box_on_image,
    save_mask_overlay,
    save_patch_crop,
    save_score_overlay,
)


@dataclass(frozen=True, slots=True)
class PatchCoreLimitsReportConfig:
    """Configuration for the synthetic PatchCore limits report."""

    output_dir: Path = Path("outputs/patchcore_limits")
    cache_path: Path = Path("data/artefacts/patchcore/limits/slot_board_bank.npz")
    synthetic_dir: Path = Path("outputs/patchcore_limits/synthetic")
    patch_size: int = 80
    stride: int = 40
    top_k: int = 3
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class PatchCoreLimitExample:
    """Report data for one synthetic limit case."""

    sample: SlotBoardSample
    score: PatchScore
    all_scores: list[PatchScore]
    mask_overlap: PatchMaskOverlap
    assets: dict[str, Path]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _sample_to_record(sample: SlotBoardSample) -> ImageManifestRecord:
    return ImageManifestRecord(
        dataset="synthetic",
        category="slot_board",
        split=sample.split,
        defect_type=sample.label,
        is_anomalous=sample.split != "train",
        image_path=sample.image_path,
        mask_path=sample.mask_path,
    )


def _build_or_load_bank(
    config: PatchCoreLimitsReportConfig,
    train_samples: list[SlotBoardSample],
    extractor: ColourTexturePatchFeatureExtractor,
) -> PatchCoreMemoryBank:
    if config.use_cache and config.cache_path.exists():
        memory_bank = load_memory_bank(config.cache_path)
        if memory_bank.feature_name == extractor.feature_name:
            return memory_bank

    memory_bank = build_patchcore_memory_bank(
        [_sample_to_record(sample) for sample in train_samples],
        extractor=extractor,
        patch_size=config.patch_size,
        stride=config.stride,
    )
    save_memory_bank(memory_bank, config.cache_path)
    return memory_bank


def _write_assets(
    *,
    example_number: int,
    sample: SlotBoardSample,
    score: PatchScore,
    scores: list[PatchScore],
    output_dir: Path,
) -> dict[str, Path]:
    prefix = f"example_{example_number}_"
    assets: dict[str, Path] = {}
    assets["score_overlay"] = save_score_overlay(
        image_path=sample.image_path,
        scores=scores,
        output_path=_asset_path(output_dir, f"{prefix}score_overlay.png"),
    )
    assets["query_box"] = draw_box_on_image(
        image_path=sample.image_path,
        box=score.query_box,
        output_path=_asset_path(output_dir, f"{prefix}query_box.png"),
    )
    assets["query_crop"] = save_patch_crop(
        image_path=sample.image_path,
        box=score.query_box,
        output_path=_asset_path(output_dir, f"{prefix}query_patch.png"),
        scale=2,
    )
    assets["mask_overlay"] = save_mask_overlay(
        image_path=sample.image_path,
        mask_path=sample.mask_path,
        output_path=_asset_path(output_dir, f"{prefix}mask_overlay.png"),
    )
    nearest = score.nearest[0]
    assets["normal_crop"] = save_patch_crop(
        image_path=nearest.metadata.source_path,
        box=nearest.metadata.box,
        output_path=_asset_path(output_dir, f"{prefix}nearest_normal_patch.png"),
        scale=2,
    )
    assets["normal_source"] = draw_box_on_image(
        image_path=nearest.metadata.source_path,
        box=nearest.metadata.box,
        output_path=_asset_path(output_dir, f"{prefix}nearest_normal_source.png"),
        colour=(30, 120, 220),
    )
    return assets


def _format_percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _render_summary_rows(examples: list[PatchCoreLimitExample]) -> str:
    rows: list[str] = []
    for example in examples:
        rows.append(
            "<tr>"
            f"<td>{html.escape(example.sample.sample_id)}</td>"
            f"<td>{html.escape(example.sample.label)}</td>"
            f"<td>{example.sample.missing_count}</td>"
            f"<td>{example.sample.severity_area}</td>"
            f"<td>{example.score.distance:.6f}</td>"
            f"<td>{_format_percentage(example.mask_overlap.mask_covered_fraction)}</td>"
            f"<td>{html.escape(example.sample.semantic_note)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_example(example: PatchCoreLimitExample, output_path: Path, example_number: int) -> str:
    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    mask_coverage = _format_percentage(example.mask_overlap.mask_covered_fraction)
    return f"""
  <section class="example">
    <h2>Example {example_number}: {html.escape(example.sample.sample_id)}</h2>
    <p>{html.escape(example.sample.semantic_note)}</p>
    <div class="grid">
      <figure>
        <img src="{rel(example.assets["score_overlay"])}" alt="Patch score overlay">
        <figcaption>Coarse patch-score overlay from the synthetic memory bank.</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["query_box"])}" alt="Top patch box">
        <figcaption>Top scored patch. Distance: {example.score.distance:.6f}</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["mask_overlay"])}" alt="Synthetic ground-truth mask">
        <figcaption>Ground-truth region from synthetic metadata.</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["normal_crop"])}" alt="Nearest nominal patch">
        <figcaption>Nearest nominal patch from the memory bank.</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["normal_source"])}" alt="Nearest nominal source image">
        <figcaption>Nearest nominal source image with source patch box.</figcaption>
      </figure>
    </div>
    <ul>
      <li>Expected component count: {example.sample.expected_count}</li>
      <li>Observed component count from metadata: {example.sample.observed_count}</li>
      <li>Missing count from metadata: {example.sample.missing_count}</li>
      <li>Severity proxy area from mask: {example.sample.severity_area}</li>
      <li>Mask covered by top patch: {mask_coverage}</li>
    </ul>
  </section>
"""


def _render_html(
    *,
    config: PatchCoreLimitsReportConfig,
    examples: list[PatchCoreLimitExample],
    output_path: Path,
    build_metadata: BuildMetadata,
) -> None:
    rows = _render_summary_rows(examples)
    sections = "\n".join(
        _render_example(example, output_path, example_number)
        for example_number, example in enumerate(examples, start=1)
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Limits Lab</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
    }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    .example {{ border-top: 2px solid #d8dee4; padding-top: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
  <h1>PatchCore Limits Lab</h1>
  <p>
    This synthetic report shows that PatchCore-style patch novelty is useful,
    but it is not a native counter, calibrated severity model, or symbolic
    logic checker.
  </p>

  {render_build_metadata_section(build_metadata)}

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>synthetic directory: <code>{html.escape(config.synthetic_dir.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(config.cache_path.as_posix())}</code></li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>colour_texture</code></li>
    </ul>
  </section>

  <section>
    <h2>Limits Overview</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Case</th>
          <th>Missing count metadata</th>
          <th>Severity area metadata</th>
          <th>Top patch score</th>
          <th>Mask covered</th>
          <th>Lesson</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <p>
      Count, severity, and semantic notes in this table come from synthetic
      metadata. PatchCore supplies patch novelty, nearest-normal provenance, and
      coarse localisation; it does not natively emit these symbolic fields.
    </p>
  </section>

{sections}
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _build_demo_card(
    output_path: Path,
    examples: list[PatchCoreLimitExample],
    *,
    build_metadata: BuildMetadata,
) -> DemoCard:
    return DemoCard(
        title="Demo 05 - PatchCore Limits Lab",
        task=(
            "Synthetic slot-board anomaly cases for PatchCore counting, severity, "
            "and logic limits."
        ),
        model=(
            "PatchCore-style nearest-neighbour memory bank with deterministic "
            "colour/texture features."
        ),
        explanation_methods=(
            "Coarse patch-score anomaly map",
            "Nearest-normal patch provenance",
            "Synthetic ground-truth mask check",
            "Metadata comparison for count, severity, and semantic labels",
        ),
        key_lesson=(
            "PatchCore can localise novelty and retrieve nearest normal evidence, "
            "but it does not natively count anomalies, calibrate severity, or emit "
            "symbolic logic statements."
        ),
        failure_mode=(
            "The detector reports patch novelty. Count, severity, and semantic "
            "meaning require extra modelling layers or domain metadata."
        ),
        intervention=(
            "Add explicit metadata-aware or logic-aware layers on top of patch novelty "
            "when the product needs counts, severity, or assembly rules."
        ),
        remaining_caveats=(
            "Synthetic toy boards are didactic, not a production benchmark.",
            "No MVTec LOCO AD comparison yet.",
            "No symbolic reasoning model is included yet.",
        ),
        report_path=output_path,
        figure_paths=(
            examples[0].assets["score_overlay"],
            examples[1].assets["score_overlay"],
            examples[2].assets["mask_overlay"],
            examples[3].assets["normal_source"],
        ),
        build_metadata=build_metadata,
    )


def build_patchcore_limits_report(config: PatchCoreLimitsReportConfig) -> Path:
    """Build a static report for the synthetic PatchCore limits lab."""

    ensure_directory(config.output_dir)
    train_samples, eval_samples = generate_slot_board_dataset(config.synthetic_dir)
    extractor = ColourTexturePatchFeatureExtractor()
    memory_bank = _build_or_load_bank(config, train_samples, extractor)
    examples: list[PatchCoreLimitExample] = []
    for example_number, sample in enumerate(eval_samples, start=1):
        scores = score_image_with_extractor(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            memory_bank=memory_bank,
            extractor=extractor,
            patch_size=config.patch_size,
            stride=config.stride,
            top_k=config.top_k,
        )
        if not scores:
            raise ValueError(f"No scores produced for {sample.sample_id}.")
        top_score = scores[0]
        overlap = measure_patch_mask_overlap(
            mask_path=sample.mask_path,
            patch_box=top_score.query_box,
            image_path=sample.image_path,
        )
        assets = _write_assets(
            example_number=example_number,
            sample=sample,
            score=top_score,
            scores=scores,
            output_dir=config.output_dir,
        )
        examples.append(
            PatchCoreLimitExample(
                sample=sample,
                score=top_score,
                all_scores=scores,
                mask_overlap=overlap,
                assets=assets,
            )
        )

    output_path = config.output_dir / "index.html"
    build_metadata = make_build_metadata(
        data_mode="synthetic",
        cache_enabled=config.use_cache,
    )
    _render_html(
        config=config,
        examples=examples,
        output_path=output_path,
        build_metadata=build_metadata,
    )
    card = _build_demo_card(output_path, examples, build_metadata=build_metadata)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
