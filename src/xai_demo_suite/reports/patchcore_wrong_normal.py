"""Static report for PatchCore learning the wrong normal."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.manifests import ImageManifestRecord
from xai_demo_suite.data.synthetic import NuisanceBoardSample, generate_nuisance_board_dataset
from xai_demo_suite.models.patchcore import (
    ColourTexturePatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchScore
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import draw_box_on_image, save_patch_crop, save_score_overlay


@dataclass(frozen=True, slots=True)
class PatchCoreWrongNormalReportConfig:
    """Configuration for the PatchCore wrong-normal report."""

    output_dir: Path = Path("outputs/patchcore_wrong_normal")
    synthetic_dir: Path = Path("outputs/patchcore_wrong_normal/synthetic")
    clean_cache_path: Path = Path("data/artefacts/patchcore/wrong_normal/clean_bank.npz")
    contaminated_cache_path: Path = Path(
        "data/artefacts/patchcore/wrong_normal/contaminated_bank.npz"
    )
    patch_size: int = 80
    stride: int = 40
    top_k: int = 3
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class WrongNormalComparison:
    """Score comparison for one query against clean and contaminated banks."""

    sample: NuisanceBoardSample
    clean_score: PatchScore
    clean_scores: list[PatchScore]
    contaminated_score: PatchScore
    contaminated_scores: list[PatchScore]
    assets: dict[str, Path]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _sample_to_record(sample: NuisanceBoardSample, *, split: str) -> ImageManifestRecord:
    return ImageManifestRecord(
        dataset="synthetic",
        category="wrong_normal_board",
        split=split,
        defect_type=sample.label,
        is_anomalous=False,
        image_path=sample.image_path,
        mask_path=sample.mask_path,
    )


def _build_or_load_bank(
    *,
    samples: list[NuisanceBoardSample],
    split: str,
    cache_path: Path,
    config: PatchCoreWrongNormalReportConfig,
    extractor: ColourTexturePatchFeatureExtractor,
) -> PatchCoreMemoryBank:
    if config.use_cache and cache_path.exists():
        memory_bank = load_memory_bank(cache_path)
        if memory_bank.feature_name == extractor.feature_name:
            return memory_bank

    memory_bank = build_patchcore_memory_bank(
        [_sample_to_record(sample, split=split) for sample in samples],
        extractor=extractor,
        patch_size=config.patch_size,
        stride=config.stride,
    )
    save_memory_bank(memory_bank, cache_path)
    return memory_bank


def _write_assets(
    *,
    output_dir: Path,
    example_number: int,
    comparison: WrongNormalComparison,
) -> dict[str, Path]:
    prefix = f"example_{example_number}_"
    sample = comparison.sample
    assets: dict[str, Path] = {}
    assets["clean_overlay"] = save_score_overlay(
        image_path=sample.image_path,
        scores=comparison.clean_scores,
        output_path=_asset_path(output_dir, f"{prefix}clean_bank_overlay.png"),
    )
    assets["contaminated_overlay"] = save_score_overlay(
        image_path=sample.image_path,
        scores=comparison.contaminated_scores,
        output_path=_asset_path(output_dir, f"{prefix}contaminated_bank_overlay.png"),
    )
    assets["tab_region"] = draw_box_on_image(
        image_path=sample.image_path,
        box=sample.tab_region,
        output_path=_asset_path(output_dir, f"{prefix}tab_region.png"),
    )
    assets["clean_patch"] = save_patch_crop(
        image_path=sample.image_path,
        box=comparison.clean_score.query_box,
        output_path=_asset_path(output_dir, f"{prefix}clean_bank_patch.png"),
        scale=2,
    )
    assets["contaminated_patch"] = save_patch_crop(
        image_path=sample.image_path,
        box=comparison.contaminated_score.query_box,
        output_path=_asset_path(output_dir, f"{prefix}contaminated_bank_patch.png"),
        scale=2,
    )
    clean_nearest = comparison.clean_score.nearest[0]
    contaminated_nearest = comparison.contaminated_score.nearest[0]
    assets["clean_source"] = draw_box_on_image(
        image_path=clean_nearest.metadata.source_path,
        box=clean_nearest.metadata.box,
        output_path=_asset_path(output_dir, f"{prefix}clean_bank_nearest_source.png"),
        colour=(40, 160, 80),
    )
    assets["contaminated_source"] = draw_box_on_image(
        image_path=contaminated_nearest.metadata.source_path,
        box=contaminated_nearest.metadata.box,
        output_path=_asset_path(output_dir, f"{prefix}contaminated_bank_nearest_source.png"),
        colour=(220, 120, 30),
    )
    return assets


def _score_sample(
    *,
    sample: NuisanceBoardSample,
    memory_bank: PatchCoreMemoryBank,
    extractor: ColourTexturePatchFeatureExtractor,
    config: PatchCoreWrongNormalReportConfig,
) -> list[PatchScore]:
    return score_image_with_extractor(
        sample_id=sample.sample_id,
        image_path=sample.image_path,
        memory_bank=memory_bank,
        extractor=extractor,
        patch_size=config.patch_size,
        stride=config.stride,
        top_k=config.top_k,
    )


def _render_rows(comparisons: list[WrongNormalComparison]) -> str:
    rows: list[str] = []
    for comparison in comparisons:
        rows.append(
            "<tr>"
            f"<td>{html.escape(comparison.sample.sample_id)}</td>"
            f"<td>{'yes' if comparison.sample.has_tab else 'no'}</td>"
            f"<td>{comparison.clean_score.distance:.6f}</td>"
            f"<td>{comparison.clean_score.nearest[0].metadata.source_image_id}</td>"
            f"<td>{comparison.contaminated_score.distance:.6f}</td>"
            f"<td>{comparison.contaminated_score.nearest[0].metadata.source_image_id}</td>"
            f"<td>{html.escape(comparison.sample.semantic_note)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_comparison(
    comparison: WrongNormalComparison,
    output_path: Path,
    example_number: int,
) -> str:
    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    clean_score_text = f"{comparison.clean_score.distance:.6f}"
    contaminated_score_text = f"{comparison.contaminated_score.distance:.6f}"
    contaminated_source_src = rel(comparison.assets["contaminated_source"])
    return f"""
  <section class="example">
    <h2>Example {example_number}: {html.escape(comparison.sample.sample_id)}</h2>
    <p>{html.escape(comparison.sample.semantic_note)}</p>
    <div class="grid">
      <figure>
        <img src="{rel(comparison.assets["tab_region"])}" alt="Expected tab region">
        <figcaption>Corner-tab region used as the nuisance location.</figcaption>
      </figure>
      <figure>
        <img src="{rel(comparison.assets["clean_overlay"])}" alt="Clean bank overlay">
        <figcaption>Clean memory bank overlay. Top score: {clean_score_text}</figcaption>
      </figure>
      <figure>
        <img src="{rel(comparison.assets["contaminated_overlay"])}" alt="Contaminated bank overlay">
        <figcaption>
          Contaminated memory bank overlay. Top score: {contaminated_score_text}
        </figcaption>
      </figure>
      <figure>
        <img src="{rel(comparison.assets["clean_source"])}" alt="Clean nearest source">
        <figcaption>Nearest normal source from the clean bank.</figcaption>
      </figure>
      <figure>
        <img src="{contaminated_source_src}" alt="Contaminated nearest source">
        <figcaption>Nearest normal source from the contaminated bank.</figcaption>
      </figure>
    </div>
  </section>
"""


def _render_html(
    *,
    config: PatchCoreWrongNormalReportConfig,
    comparisons: list[WrongNormalComparison],
    output_path: Path,
) -> None:
    rows = _render_rows(comparisons)
    contaminated_cache = html.escape(config.contaminated_cache_path.as_posix())
    sections = "\n".join(
        _render_comparison(comparison, output_path, index)
        for index, comparison in enumerate(comparisons, start=1)
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Learns the Wrong Normal</title>
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
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>PatchCore Learns the Wrong Normal</h1>
  <p>
    This report compares a clean PatchCore-style memory bank with a nominal set
    contaminated by a corner acquisition tab. It shows how the memory bank can
    encode nuisance as normality and distort explanations.
  </p>

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>synthetic directory: <code>{html.escape(config.synthetic_dir.as_posix())}</code></li>
      <li>clean cache: <code>{html.escape(config.clean_cache_path.as_posix())}</code></li>
      <li>contaminated cache: <code>{contaminated_cache}</code></li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>colour_texture</code></li>
    </ul>
  </section>

  <section>
    <h2>Clean Bank vs Contaminated Bank</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Has tab</th>
          <th>Clean bank top score</th>
          <th>Clean nearest source</th>
          <th>Contaminated bank top score</th>
          <th>Contaminated nearest source</th>
          <th>Lesson</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <p>
      The fix is not a new threshold. The fix is to clean the nominal pipeline,
      rebuild the memory bank, and then re-check the explanation.
    </p>
  </section>

{sections}
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _build_demo_card(output_path: Path, comparisons: list[WrongNormalComparison]) -> DemoCard:
    return DemoCard(
        title="Demo 04 - PatchCore Learns the Wrong Normal",
        task="Synthetic normal-set contamination for PatchCore-style anomaly detection.",
        model="Two PatchCore-style memory banks with deterministic colour/texture features.",
        explanation_methods=(
            "Clean versus contaminated anomaly maps",
            "Nearest-normal provenance comparison",
            "Nuisance-region counter-test",
        ),
        key_lesson=(
            "If nuisance enters the nominal set, PatchCore can encode it as normality "
            "and produce misleading explanations."
        ),
        failure_mode="The contaminated bank treats a corner acquisition tab as normal.",
        intervention="Clean the nominal pipeline and rebuild the memory bank.",
        remaining_caveats=(
            "Synthetic didactic board, not a production acquisition-shift benchmark.",
            "No temporal drift monitoring yet.",
            "No real nuisance-contaminated industrial dataset yet.",
        ),
        report_path=output_path,
        figure_paths=(
            comparisons[0].assets["clean_overlay"],
            comparisons[0].assets["contaminated_overlay"],
            comparisons[1].assets["clean_source"],
            comparisons[1].assets["contaminated_source"],
        ),
    )


def build_patchcore_wrong_normal_report(config: PatchCoreWrongNormalReportConfig) -> Path:
    """Build the PatchCore wrong-normal report."""

    ensure_directory(config.output_dir)
    clean_train, contaminated_train, query_samples = generate_nuisance_board_dataset(
        config.synthetic_dir
    )
    extractor = ColourTexturePatchFeatureExtractor()
    clean_bank = _build_or_load_bank(
        samples=clean_train,
        split="train_clean",
        cache_path=config.clean_cache_path,
        config=config,
        extractor=extractor,
    )
    contaminated_bank = _build_or_load_bank(
        samples=contaminated_train,
        split="train_contaminated",
        cache_path=config.contaminated_cache_path,
        config=config,
        extractor=extractor,
    )

    comparisons: list[WrongNormalComparison] = []
    for sample in query_samples:
        clean_scores = _score_sample(
            sample=sample,
            memory_bank=clean_bank,
            extractor=extractor,
            config=config,
        )
        contaminated_scores = _score_sample(
            sample=sample,
            memory_bank=contaminated_bank,
            extractor=extractor,
            config=config,
        )
        comparisons.append(
            WrongNormalComparison(
                sample=sample,
                clean_score=clean_scores[0],
                clean_scores=clean_scores,
                contaminated_score=contaminated_scores[0],
                contaminated_scores=contaminated_scores,
                assets={},
            )
        )

    enriched: list[WrongNormalComparison] = []
    for index, comparison in enumerate(comparisons, start=1):
        enriched.append(
            WrongNormalComparison(
                sample=comparison.sample,
                clean_score=comparison.clean_score,
                clean_scores=comparison.clean_scores,
                contaminated_score=comparison.contaminated_score,
                contaminated_scores=comparison.contaminated_scores,
                assets=_write_assets(
                    output_dir=config.output_dir,
                    example_number=index,
                    comparison=comparison,
                ),
            )
        )

    output_path = config.output_dir / "index.html"
    _render_html(config=config, comparisons=enriched, output_path=output_path)
    card = _build_demo_card(output_path, enriched)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
