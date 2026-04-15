"""Static report for PatchCore logical anomaly limitations."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.manifests import (
    ImageManifestRecord,
    filter_manifest_records,
    load_image_manifest,
)
from xai_demo_suite.data.synthetic import generate_slot_board_dataset
from xai_demo_suite.evaluate.localisation import PatchMaskOverlap, measure_patch_mask_overlap
from xai_demo_suite.models.patchcore import (
    ColourTexturePatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchScore
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.reports.patchcore_synthetic_helpers import (
    SyntheticPatchCoreExample,
    build_or_load_synthetic_bank,
    build_synthetic_patchcore_examples,
    format_percentage,
    relative_path,
)
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import (
    draw_box_on_image,
    save_mask_overlay,
    save_patch_crop,
    save_score_overlay,
)


@dataclass(frozen=True, slots=True)
class PatchCoreLogicReportConfig:
    """Configuration for the PatchCore logic report."""

    output_dir: Path = Path("outputs/patchcore_logic")
    cache_path: Path = Path("data/artefacts/patchcore/logic/logic_bank.npz")
    loco_cache_path: Path = Path("data/artefacts/patchcore/loco/juice_bottle_logic_bank.npz")
    manifest_path: Path = Path("data/processed/mvtec_loco_ad/juice_bottle/manifest.jsonl")
    synthetic_dir: Path = Path("outputs/patchcore_logic/synthetic")
    patch_size: int = 80
    stride: int = 40
    loco_patch_size: int = 240
    loco_stride: int = 160
    loco_max_train: int = 40
    top_k: int = 3
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class RealPatchCoreLogicExample:
    """Report data for one real MVTec LOCO logical/structural example."""

    record: ImageManifestRecord
    score: PatchScore
    all_scores: list[PatchScore]
    mask_overlap: PatchMaskOverlap
    assets: dict[str, Path]
    rule_statement: str
    interpretation: str


def _logic_eval_examples(
    examples: list[SyntheticPatchCoreExample],
) -> list[SyntheticPatchCoreExample]:
    preferred = {"missing_one", "logic_swap"}
    return [example for example in examples if example.sample.sample_id in preferred]


def _render_rows(examples: list[SyntheticPatchCoreExample]) -> str:
    rows: list[str] = []
    for example in examples:
        native_statement = (
            "Top patch novelty and nearest normal patch"
            if example.sample.sample_id == "logic_swap"
            else "Top patch novelty around missing component"
        )
        required_statement = (
            "Two slots contain the wrong component identities"
            if example.sample.sample_id == "logic_swap"
            else "Slot 3 is empty"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(example.sample.sample_id)}</td>"
            f"<td>{html.escape(example.sample.label)}</td>"
            f"<td>{example.score.distance:.6f}</td>"
            f"<td>{format_percentage(example.mask_overlap.mask_covered_fraction)}</td>"
            f"<td>{html.escape(native_statement)}</td>"
            f"<td>{html.escape(required_statement)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_real_rows(examples: list[RealPatchCoreLogicExample]) -> str:
    rows: list[str] = []
    for example in examples:
        native_statement = "Top patch novelty, nearest normal source patch, and mask overlap"
        rows.append(
            "<tr>"
            f"<td>{html.escape(example.record.defect_type)}</td>"
            f"<td>{html.escape(example.record.image_path.name)}</td>"
            f"<td>{example.score.distance:.6f}</td>"
            f"<td>{format_percentage(example.mask_overlap.mask_covered_fraction)}</td>"
            f"<td>{html.escape(native_statement)}</td>"
            f"<td>{html.escape(example.rule_statement)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_example(example: SyntheticPatchCoreExample, output_path: Path) -> str:
    def rel(path: Path) -> str:
        return html.escape(relative_path(path, output_path.parent))

    if example.sample.sample_id == "logic_swap":
        interpretation = (
            "PatchCore can highlight unusual local patches, but the claim that the "
            "red and blue components are in the wrong semantic positions is a rule "
            "outside the patch-distance model."
        )
    else:
        interpretation = (
            "A missing component is locally visible, but the statement that slot 3 "
            "is empty still comes from slot metadata or a component model."
        )
    return f"""
  <section class="example">
    <h2>{html.escape(example.sample.sample_id)}</h2>
    <p>{html.escape(example.sample.semantic_note)}</p>
    <div class="grid">
      <figure>
        <img src="{rel(example.assets["score_overlay"])}" alt="Patch score overlay">
        <figcaption>Patch-score overlay. Top score: {example.score.distance:.6f}</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["query_box"])}" alt="Top patch box">
        <figcaption>Top novelty patch selected by the detector.</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["mask_overlay"])}" alt="Logic mask overlay">
        <figcaption>Synthetic mask for the changed local regions.</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["normal_source"])}" alt="Nearest normal source">
        <figcaption>Nearest nominal source patch used as provenance.</figcaption>
      </figure>
    </div>
    <p>{html.escape(interpretation)}</p>
  </section>
"""


def _render_real_example(example: RealPatchCoreLogicExample, output_path: Path) -> str:
    def rel(path: Path) -> str:
        return html.escape(relative_path(path, output_path.parent))

    nearest = example.score.nearest[0]
    heading = f"{example.record.defect_type}: {example.record.image_path.name}"
    return f"""
  <section class="example">
    <h2>{html.escape(heading)}</h2>
    <p>{html.escape(example.interpretation)}</p>
    <div class="grid">
      <figure>
        <img src="{rel(example.assets["score_overlay"])}" alt="Patch score overlay">
        <figcaption>Patch-score overlay. Top score: {example.score.distance:.6f}</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["query_box"])}" alt="Top patch box">
        <figcaption>Top novelty patch selected by the detector.</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["mask_overlay"])}" alt="Ground-truth mask overlay">
        <figcaption>MVTec LOCO ground-truth region. Mask covered by top patch:
        {format_percentage(example.mask_overlap.mask_covered_fraction)}</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["normal_source"])}" alt="Nearest normal source">
        <figcaption>Nearest normal source patch:
        {html.escape(nearest.metadata.source_image_id)}</figcaption>
      </figure>
    </div>
  </section>
"""


def _render_html(
    *,
    config: PatchCoreLogicReportConfig,
    examples: list[SyntheticPatchCoreExample],
    output_path: Path,
) -> None:
    rows = _render_rows(examples)
    sections = "\n".join(_render_example(example, output_path) for example in examples)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Logical Anomaly Limits</title>
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
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
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
  <h1>PatchCore Logical Anomaly Limits</h1>
  <p>
    Demo 07 separates local novelty from logical understanding. The synthetic
    board gives PatchCore-style scoring a fair local anomaly signal, then asks
    for statements such as slot occupancy and component identity.
  </p>

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>synthetic directory: <code>{html.escape(config.synthetic_dir.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(config.cache_path.as_posix())}</code></li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>colour_texture</code></li>
      <li>MVTec LOCO AD comparison: not sourced yet.</li>
    </ul>
  </section>

  <section>
    <h2>Patch Novelty Versus Rule Statements</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Case</th>
          <th>Top patch score</th>
          <th>Mask covered</th>
          <th>Native PatchCore-style output</th>
          <th>Rule statement the product may need</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

{sections}

  <section>
    <h2>Lesson</h2>
    <p>
      PatchCore-style retrieval makes local evidence inspectable, but a logical
      anomaly product needs additional structure: slot definitions, component
      identity, expected relations, or a logic-aware model. This is why MVTec
      LOCO AD remains an important follow-up dataset.
    </p>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _render_real_html(
    *,
    config: PatchCoreLogicReportConfig,
    examples: list[RealPatchCoreLogicExample],
    train_count: int,
    output_path: Path,
) -> None:
    rows = _render_real_rows(examples)
    sections = "\n".join(_render_real_example(example, output_path) for example in examples)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Logical Anomaly Limits on MVTec LOCO</title>
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
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
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
  <h1>PatchCore Logical Anomaly Limits on MVTec LOCO</h1>
  <p>
    Demo 07 now uses real MVTec LOCO AD juice-bottle images. The detector sees
    local patch novelty and nearest normal evidence; the product-level question
    asks whether the bottle satisfies a semantic packaging rule.
  </p>

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>manifest: <code>{html.escape(config.manifest_path.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(config.loco_cache_path.as_posix())}</code></li>
      <li>nominal training images used: {train_count}</li>
      <li>patch size: {config.loco_patch_size}, stride: {config.loco_stride},
      top-k: {config.top_k}</li>
      <li>feature extractor: <code>colour_texture</code></li>
    </ul>
  </section>

  <section>
    <h2>Patch Novelty Versus Rule Statements</h2>
    <table>
      <thead>
        <tr>
          <th>LOCO case</th>
          <th>Image</th>
          <th>Top patch score</th>
          <th>Mask covered</th>
          <th>Native PatchCore-style output</th>
          <th>Rule statement the product may need</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

{sections}

  <section>
    <h2>Lesson</h2>
    <p>
      The real LOCO cases make the boundary sharper than the synthetic proxy:
      PatchCore-style retrieval can expose where the image differs from normal
      bottles, but it still does not natively know that a front label is required
      or that packaging text and graphics must satisfy a product rule.
    </p>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _build_demo_card(output_path: Path, examples: list[SyntheticPatchCoreExample]) -> DemoCard:
    return DemoCard(
        title="Demo 07 - PatchCore Logical Anomaly Limits",
        task="Synthetic slot-board comparison of local novelty and logical rule violations.",
        model="PatchCore-style nearest-neighbour memory bank with colour/texture patch features.",
        explanation_methods=(
            "Patch-score anomaly overlay",
            "Nearest-normal patch provenance",
            "Synthetic rule-statement comparison",
            "Structural versus logical case contrast",
        ),
        key_lesson=(
            "PatchCore can localise novelty, but it does not natively understand "
            "slot rules or component identities."
        ),
        failure_mode=(
            "The detector emits patch distances, not statements such as slot 3 is "
            "empty or two components are swapped."
        ),
        intervention=(
            "Add component detection, slot metadata, or logic-aware reasoning when "
            "the product needs rule-level explanations."
        ),
        remaining_caveats=(
            "Synthetic proxy, not an MVTec LOCO AD comparison yet.",
            "No symbolic or component-aware baseline yet.",
            "No real assembly-rule dataset yet.",
        ),
        report_path=output_path,
        figure_paths=(
            examples[0].assets["score_overlay"],
            examples[0].assets["normal_source"],
            examples[1].assets["score_overlay"],
            examples[1].assets["mask_overlay"],
        ),
    )


def _build_real_demo_card(
    output_path: Path,
    examples: list[RealPatchCoreLogicExample],
) -> DemoCard:
    return DemoCard(
        title="Demo 07 - PatchCore Logical Anomaly Limits",
        task=(
            "MVTec LOCO AD juice-bottle comparison of local novelty and logical "
            "packaging-rule violations."
        ),
        model="PatchCore-style nearest-neighbour memory bank with colour/texture patch features.",
        explanation_methods=(
            "Patch-score anomaly overlay",
            "Nearest-normal patch provenance",
            "MVTec LOCO mask overlap check",
            "Structural versus logical case contrast",
        ),
        key_lesson=(
            "Real LOCO examples show that local novelty is inspectable, but product "
            "rules still require semantic or component-aware logic."
        ),
        failure_mode=(
            "The detector emits patch distances and nearest normal patches, not "
            "statements such as required front label missing."
        ),
        intervention=(
            "Use PatchCore as a provenance-rich triage layer, then add component, "
            "OCR, template, or logic-aware checks for packaging rules."
        ),
        remaining_caveats=(
            "Uses one LOCO category and deterministic local patch features.",
            "No component-aware or OCR comparator yet.",
            "No pretrained multi-scale PatchCore path yet.",
        ),
        report_path=output_path,
        figure_paths=(
            examples[0].assets["score_overlay"],
            examples[0].assets["mask_overlay"],
            examples[1].assets["score_overlay"],
            examples[1].assets["normal_source"],
        ),
    )


def _build_synthetic_logic_report(config: PatchCoreLogicReportConfig) -> Path:
    """Build the synthetic fallback report for fresh clones."""

    ensure_directory(config.output_dir)
    train_samples, eval_samples = generate_slot_board_dataset(config.synthetic_dir)
    extractor = ColourTexturePatchFeatureExtractor()
    memory_bank = build_or_load_synthetic_bank(
        train_samples=train_samples,
        extractor=extractor,
        cache_path=config.cache_path,
        use_cache=config.use_cache,
        patch_size=config.patch_size,
        stride=config.stride,
        category="slot_board_logic",
    )
    all_examples = build_synthetic_patchcore_examples(
        eval_samples=eval_samples,
        memory_bank=memory_bank,
        extractor=extractor,
        output_dir=config.output_dir,
        patch_size=config.patch_size,
        stride=config.stride,
        top_k=config.top_k,
    )
    examples = _logic_eval_examples(all_examples)
    if len(examples) != 2:
        raise ValueError("Logic report requires missing_one and logic_swap examples.")
    output_path = config.output_dir / "index.html"
    _render_html(config=config, examples=examples, output_path=output_path)
    save_demo_card(_build_demo_card(output_path, examples), config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path


def _build_or_load_loco_bank(
    *,
    config: PatchCoreLogicReportConfig,
    train_records: list[ImageManifestRecord],
    extractor: ColourTexturePatchFeatureExtractor,
) -> PatchCoreMemoryBank:
    if config.use_cache and config.loco_cache_path.exists():
        memory_bank = load_memory_bank(config.loco_cache_path)
        if memory_bank.feature_name == extractor.feature_name:
            return memory_bank

    memory_bank = build_patchcore_memory_bank(
        train_records,
        extractor=extractor,
        patch_size=config.loco_patch_size,
        stride=config.loco_stride,
    )
    save_memory_bank(memory_bank, config.loco_cache_path)
    return memory_bank


def _select_loco_record(
    records: list[ImageManifestRecord],
    defect_type: str,
) -> ImageManifestRecord:
    candidates = [
        record
        for record in filter_manifest_records(records, split="test", defect_type=defect_type)
        if record.mask_path is not None
    ]
    if not candidates:
        raise ValueError(f"No MVTec LOCO test record with a mask for {defect_type}.")
    return candidates[0]


def _write_loco_assets(
    *,
    record: ImageManifestRecord,
    score: PatchScore,
    scores: list[PatchScore],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    if record.mask_path is None:
        raise ValueError(f"Record {record.sample_id} has no mask path.")

    asset_dir = output_dir / "assets"
    assets: dict[str, Path] = {}
    assets["score_overlay"] = save_score_overlay(
        image_path=record.image_path,
        scores=scores,
        output_path=asset_dir / f"{prefix}_score_overlay.png",
    )
    assets["query_box"] = draw_box_on_image(
        image_path=record.image_path,
        box=score.query_box,
        output_path=asset_dir / f"{prefix}_query_box.png",
    )
    assets["mask_overlay"] = save_mask_overlay(
        image_path=record.image_path,
        mask_path=record.mask_path,
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


def _build_loco_example(
    *,
    record: ImageManifestRecord,
    memory_bank: PatchCoreMemoryBank,
    extractor: ColourTexturePatchFeatureExtractor,
    config: PatchCoreLogicReportConfig,
    rule_statement: str,
    interpretation: str,
) -> RealPatchCoreLogicExample:
    scores = score_image_with_extractor(
        sample_id=record.sample_id,
        image_path=record.image_path,
        memory_bank=memory_bank,
        extractor=extractor,
        patch_size=config.loco_patch_size,
        stride=config.loco_stride,
        top_k=config.top_k,
    )
    if not scores:
        raise ValueError(f"No scores produced for {record.sample_id}.")
    top_score = scores[0]
    if record.mask_path is None:
        raise ValueError(f"Record {record.sample_id} has no mask path.")
    overlap = measure_patch_mask_overlap(
        mask_path=record.mask_path,
        patch_box=top_score.query_box,
        image_path=record.image_path,
    )
    prefix = f"loco_{record.defect_type}_{record.image_path.stem}"
    assets = _write_loco_assets(
        record=record,
        score=top_score,
        scores=scores,
        output_dir=config.output_dir,
        prefix=prefix,
    )
    return RealPatchCoreLogicExample(
        record=record,
        score=top_score,
        all_scores=scores,
        mask_overlap=overlap,
        assets=assets,
        rule_statement=rule_statement,
        interpretation=interpretation,
    )


def _build_real_logic_report(config: PatchCoreLogicReportConfig) -> Path:
    """Build the real MVTec LOCO path for Demo 07."""

    ensure_directory(config.output_dir)
    records = load_image_manifest(config.manifest_path)
    train_records = filter_manifest_records(records, split="train", defect_type="good")[
        : config.loco_max_train
    ]
    if not train_records:
        raise ValueError("MVTec LOCO Demo 07 requires nominal train records.")

    extractor = ColourTexturePatchFeatureExtractor()
    memory_bank = _build_or_load_loco_bank(
        config=config,
        train_records=train_records,
        extractor=extractor,
    )
    logical_record = _select_loco_record(records, "logical_anomalies")
    structural_record = _select_loco_record(records, "structural_anomalies")
    examples = [
        _build_loco_example(
            record=logical_record,
            memory_bank=memory_bank,
            extractor=extractor,
            config=config,
            rule_statement="Required front label or packaging element is missing.",
            interpretation=(
                "The missing label is a logical packaging failure: the local region is "
                "visible, but the product-level rule is that this front label must exist."
            ),
        ),
        _build_loco_example(
            record=structural_record,
            memory_bank=memory_bank,
            extractor=extractor,
            config=config,
            rule_statement="Visible structural damage or foreign material on the bottle.",
            interpretation=(
                "The structural case is closer to what PatchCore naturally handles: a "
                "local visual defect that differs from nominal source patches."
            ),
        ),
    ]
    output_path = config.output_dir / "index.html"
    _render_real_html(
        config=config,
        examples=examples,
        train_count=len(train_records),
        output_path=output_path,
    )
    save_demo_card(_build_real_demo_card(output_path, examples), config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path


def build_patchcore_logic_report(config: PatchCoreLogicReportConfig) -> Path:
    """Build Demo 07, using real MVTec LOCO data when available."""

    if config.manifest_path.exists():
        return _build_real_logic_report(config)
    return _build_synthetic_logic_report(config)
