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
from xai_demo_suite.models.patchcore import (
    PatchFeatureExtractor,
    TorchvisionBackbonePatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchCoreMemoryBank, PatchScore
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import draw_box_on_image, save_patch_crop


@dataclass(frozen=True, slots=True)
class PatchCoreBottleReportConfig:
    """Configuration for the first static PatchCore bottle report."""

    manifest_path: Path = Path("data/processed/mvtec_ad/bottle/manifest.jsonl")
    output_dir: Path = Path("outputs/patchcore_bottle")
    cache_path: Path = Path("data/artefacts/patchcore/bottle/report_resnet18_bank.npz")
    max_train: int = 2
    test_index: int = 0
    patch_size: int = 128
    stride: int = 128
    top_k: int = 3
    input_size: int = 64
    batch_size: int = 8
    use_cache: bool = True


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


def _select_query_record(config: PatchCoreBottleReportConfig) -> ImageManifestRecord:
    records = load_image_manifest(config.manifest_path)
    query_records = filter_manifest_records(records, split="test", is_anomalous=True)
    if not query_records:
        raise ValueError("No anomalous test records found for MVTec AD bottle.")
    try:
        return query_records[config.test_index]
    except IndexError as exc:
        raise ValueError(
            f"test_index {config.test_index} is out of range for {len(query_records)} records."
        ) from exc


def _write_assets(
    *,
    score: PatchScore,
    output_dir: Path,
) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    assets["query_box"] = draw_box_on_image(
        image_path=score.image_path,
        box=score.query_box,
        output_path=_asset_path(output_dir, "query_box.png"),
    )
    assets["query_crop"] = save_patch_crop(
        image_path=score.image_path,
        box=score.query_box,
        output_path=_asset_path(output_dir, "query_patch.png"),
        scale=3,
    )
    for index, neighbour in enumerate(score.nearest, start=1):
        assets[f"normal_crop_{index}"] = save_patch_crop(
            image_path=neighbour.metadata.source_path,
            box=neighbour.metadata.box,
            output_path=_asset_path(output_dir, f"normal_patch_{index}.png"),
            scale=3,
        )
        assets[f"normal_box_{index}"] = draw_box_on_image(
            image_path=neighbour.metadata.source_path,
            box=neighbour.metadata.box,
            output_path=_asset_path(output_dir, f"normal_source_{index}.png"),
            colour=(30, 120, 220),
        )
    return assets


def _render_html(
    *,
    config: PatchCoreBottleReportConfig,
    score: PatchScore,
    all_scores: list[PatchScore],
    assets: dict[str, Path],
    output_path: Path,
) -> None:
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
    neighbour_figures = "\n".join(
        f"""
      <figure>
        <img src="{rel(assets[f"normal_crop_{index}"])}" alt="Nearest normal patch {index}">
        <figcaption>Nearest normal patch {index}; distance {score.nearest[index - 1].distance:.6f}</figcaption>
      </figure>
      <figure>
        <img src="{rel(assets[f"normal_box_{index}"])}" alt="Source image for nearest normal patch {index}">
        <figcaption>Full source image with patch box {index}.</figcaption>
      </figure>
      """
        for index in range(1, len(score.nearest) + 1)
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Bottle Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; align-items: start; }}
    figure {{ margin: 0; border: 1px solid #d8dee4; padding: 10px; background: #fff; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ font-size: 13px; color: #52606d; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; vertical-align: top; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>PatchCore Bottle Report</h1>
  <p>This report is generated from package code. It shows nearest-normal patch provenance for one MVTec AD bottle anomaly candidate.</p>

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>manifest: <code>{html.escape(config.manifest_path.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(config.cache_path.as_posix())}</code></li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>torchvision_resnet18</code>, random weights</li>
    </ul>
  </section>

  <section>
    <h2>Top Scored Patch</h2>
    <div class="grid">
      <figure>
        <img src="{query_box_src}" alt="Input image with top scored patch box">
        <figcaption>Input image with the top scored patch highlighted.</figcaption>
      </figure>
      <figure>
        <img src="{query_crop_src}" alt="Top scored query patch crop">
        <figcaption>Top scored query patch. Distance: {score.distance:.6f}</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2>Nearest Normal Patch Evidence</h2>
    <div class="grid">
      {neighbour_figures}
    </div>
  </section>

  <section>
    <h2>Distance Summary</h2>
    <table>
      <thead><tr><th>Rank</th><th>Source image id</th><th>Distance</th><th>Source box</th><th>Source path</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Top Query Patch Distances</h2>
    <ol>{top_scores}</ol>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def build_patchcore_bottle_report(
    config: PatchCoreBottleReportConfig,
    extractor: PatchFeatureExtractor | None = None,
) -> Path:
    """Build a static HTML report for one MVTec AD bottle example."""

    ensure_directory(config.output_dir)
    extractor = extractor or _build_default_extractor(config)
    memory_bank = _build_or_load_bank(config, extractor)
    query_record = _select_query_record(config)
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
        raise ValueError("No query patch scores were produced.")

    top_score = scores[0]
    assets = _write_assets(score=top_score, output_dir=config.output_dir)
    output_path = config.output_dir / "index.html"
    _render_html(
        config=config,
        score=top_score,
        all_scores=scores,
        assets=assets,
        output_path=output_path,
    )
    return output_path
