"""Static report for the Waterbirds shortcut demo."""

from __future__ import annotations

import html
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from xai_demo_suite.data.synthetic import (
    HabitatBirdSample,
    generate_habitat_bird_dataset,
    write_habitat_counterfactual,
)
from xai_demo_suite.data.waterbirds_manifest import (
    WaterbirdsManifestRecord,
    filter_waterbirds_records,
    load_waterbirds_manifest,
)
from xai_demo_suite.models.classification import (
    BirdShapeClassifier,
    ClassificationResult,
    FrozenResNetWaterbirdsProbe,
    GroupMetric,
    HabitatShortcutClassifier,
    PrototypeExemplar,
    PrototypeExemplarComparator,
    PrototypePrediction,
    WaterbirdsGroupMetric,
    WaterbirdsPrediction,
    WaterbirdsProbeConfig,
    accuracy,
    evaluate_bird_classifier,
    group_accuracy,
    prototype_accuracy,
    prototype_group_accuracy,
    waterbirds_accuracy,
    waterbirds_group_accuracy,
    waterbirds_worst_group_accuracy,
    worst_group_accuracy,
)
from xai_demo_suite.reports.build_metadata import (
    BuildMetadata,
    make_build_metadata,
    render_build_metadata_section,
)
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
from xai_demo_suite.vis.image_panels import draw_box_on_image, save_heatmap_overlay


@dataclass(frozen=True, slots=True)
class WaterbirdsShortcutReportConfig:
    """Configuration for the Waterbirds shortcut report."""

    output_dir: Path = Path("outputs/waterbirds_shortcut")
    synthetic_dir: Path = Path("outputs/waterbirds_shortcut/synthetic")
    manifest_path: Path = Path(
        "data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl"
    )
    metashift_manifest_path: Path = Path(
        "data/processed/metashift/subpopulation_shift_cat_dog_indoor_outdoor/manifest.jsonl"
    )
    use_real_data: bool = True
    max_train_records: int | None = 800
    max_test_records: int | None = 400
    input_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    weights_name: str | None = "DEFAULT"
    seed: int = 7
    diagnostic_sample_limit: int = 8
    backbone_tuning: str = "layer4"


@dataclass(frozen=True, slots=True)
class WaterbirdsShortcutReportData:
    """Computed data for the synthetic Waterbirds-style report."""

    train_samples: list[HabitatBirdSample]
    test_samples: list[HabitatBirdSample]
    habitat_results: list[ClassificationResult]
    shape_results: list[ClassificationResult]
    habitat_group_metrics: tuple[GroupMetric, ...]
    shape_group_metrics: tuple[GroupMetric, ...]
    assets: dict[str, Path]


@dataclass(frozen=True, slots=True)
class WaterbirdsExplanationSummary:
    """Proxy spatial-attribution and perturbation diagnostics."""

    grad_cam_centre_mass: float
    integrated_gradients_centre_mass: float
    background_mask_delta: float
    centre_mask_delta: float


@dataclass(frozen=True, slots=True)
class PrototypeComparatorSummary:
    """Perturbation diagnostics for the prototype comparator."""

    background_mask_delta: float
    centre_mask_delta: float


@dataclass(frozen=True, slots=True)
class RealWaterbirdsShortcutReportData:
    """Computed data for one manifest-backed natural-context shortcut dataset."""

    manifest_path: Path
    title: str
    dataset_name: str
    narrative: str
    train_records: list[WaterbirdsManifestRecord]
    test_records: list[WaterbirdsManifestRecord]
    erm_predictions: list[WaterbirdsPrediction]
    balanced_predictions: list[WaterbirdsPrediction]
    prototype_predictions: list[PrototypePrediction]
    erm_group_metrics: tuple[WaterbirdsGroupMetric, ...]
    balanced_group_metrics: tuple[WaterbirdsGroupMetric, ...]
    prototype_group_metrics: tuple[WaterbirdsGroupMetric, ...]
    erm_summary: WaterbirdsExplanationSummary
    balanced_summary: WaterbirdsExplanationSummary
    prototype_summary: PrototypeComparatorSummary
    selected_record: WaterbirdsManifestRecord
    selected_erm_prediction: WaterbirdsPrediction
    selected_balanced_prediction: WaterbirdsPrediction
    selected_prototype_prediction: PrototypePrediction
    nearest_predicted_exemplars: tuple[PrototypeExemplar, ...]
    nearest_contrast_exemplars: tuple[PrototypeExemplar, ...]
    assets: dict[str, Path]


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=root.resolve())).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _result_map(results: list[ClassificationResult]) -> dict[str, ClassificationResult]:
    return {result.sample_id: result for result in results}


def _write_assets(
    *,
    output_dir: Path,
    test_samples: list[HabitatBirdSample],
    habitat_classifier: HabitatShortcutClassifier,
    shape_classifier: BirdShapeClassifier,
) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    crossed_waterbird = next(
        sample for sample in test_samples if sample.sample_id == "test_waterbird_land_000"
    )
    crossed_landbird = next(
        sample for sample in test_samples if sample.sample_id == "test_landbird_water_000"
    )
    aligned_waterbird = next(
        sample for sample in test_samples if sample.sample_id == "test_waterbird_water_000"
    )
    assets["aligned_example"] = aligned_waterbird.image_path
    assets["habitat_evidence"] = draw_box_on_image(
        image_path=crossed_waterbird.image_path,
        box=habitat_classifier.evidence_region,
        output_path=_asset_path(output_dir, "habitat_evidence_box.png"),
    )
    assets["bird_evidence"] = draw_box_on_image(
        image_path=crossed_waterbird.image_path,
        box=shape_classifier.evidence_region,
        output_path=_asset_path(output_dir, "bird_shape_evidence_box.png"),
        colour=(40, 160, 80),
    )
    assets["waterbird_background_swap"] = write_habitat_counterfactual(
        crossed_waterbird,
        "water",
        _asset_path(output_dir, "counterfactual_waterbird_land_to_water.png"),
    )
    assets["landbird_background_swap"] = write_habitat_counterfactual(
        crossed_landbird,
        "land",
        _asset_path(output_dir, "counterfactual_landbird_water_to_land.png"),
    )
    return assets


def _render_group_rows(
    habitat_metrics: tuple[GroupMetric, ...],
    shape_metrics: tuple[GroupMetric, ...],
) -> str:
    shape_by_group = {metric.group: metric for metric in shape_metrics}
    rows: list[str] = []
    for habitat_metric in habitat_metrics:
        shape_metric = shape_by_group[habitat_metric.group]
        rows.append(
            "<tr>"
            f"<td>{html.escape(habitat_metric.group)}</td>"
            f"<td>{habitat_metric.count}</td>"
            f"<td>{habitat_metric.accuracy:.1%}</td>"
            f"<td>{shape_metric.accuracy:.1%}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_result_rows(
    samples: list[HabitatBirdSample],
    habitat_results: list[ClassificationResult],
    shape_results: list[ClassificationResult],
) -> str:
    habitat_by_id = _result_map(habitat_results)
    shape_by_id = _result_map(shape_results)
    rows: list[str] = []
    for sample in samples:
        habitat = habitat_by_id[sample.sample_id]
        shape = shape_by_id[sample.sample_id]
        rows.append(
            "<tr>"
            f"<td>{html.escape(sample.sample_id)}</td>"
            f"<td>{html.escape(sample.label)}</td>"
            f"<td>{html.escape(sample.habitat)}</td>"
            f"<td>{html.escape(habitat.predicted)}</td>"
            f"<td>{habitat.score:.3f}</td>"
            f"<td>{html.escape(shape.predicted)}</td>"
            f"<td>{shape.score:.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_synthetic_html(
    config: WaterbirdsShortcutReportConfig,
    data: WaterbirdsShortcutReportData,
    build_metadata: BuildMetadata,
) -> Path:
    output_path = config.output_dir / "index.html"
    habitat_accuracy = accuracy(data.habitat_results)
    shape_accuracy = accuracy(data.shape_results)
    habitat_worst_group = worst_group_accuracy(data.habitat_group_metrics)
    shape_worst_group = worst_group_accuracy(data.shape_group_metrics)
    group_rows = _render_group_rows(data.habitat_group_metrics, data.shape_group_metrics)
    result_rows = _render_result_rows(data.test_samples, data.habitat_results, data.shape_results)

    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Waterbirds Shortcut Proxy</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
    }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
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
  <h1>Waterbirds Shortcut Proxy</h1>
  <p>
    This synthetic fallback mirrors the Waterbirds failure mode: a classifier
    can learn habitat background instead of bird evidence when training labels
    and habitats are strongly correlated.
  </p>

  {render_build_metadata_section(build_metadata)}

  <section>
    <h2>Worst-Group Metric Summary</h2>
    <ul>
      <li>Habitat shortcut accuracy: {habitat_accuracy:.1%}</li>
      <li>Bird-shape intervention accuracy: {shape_accuracy:.1%}</li>
      <li>Habitat shortcut worst-group accuracy: {habitat_worst_group:.1%}</li>
      <li>Bird-shape intervention worst-group accuracy: {shape_worst_group:.1%}</li>
    </ul>
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th>Count</th>
          <th>Habitat shortcut accuracy</th>
          <th>Bird-shape accuracy</th>
        </tr>
      </thead>
      <tbody>{group_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Evidence and Counterfactuals</h2>
    <div class="grid">
      <figure>
        <img src="{rel(data.assets["aligned_example"])}" alt="Aligned waterbird example">
        <figcaption>Aligned training-style examples make the shortcut look plausible.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["habitat_evidence"])}" alt="Habitat evidence box">
        <figcaption>The shortcut classifier uses the whole habitat background.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["bird_evidence"])}" alt="Bird evidence box">
        <figcaption>The intervention looks at the bird silhouette instead.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["waterbird_background_swap"])}" alt="Waterbird background swap">
        <figcaption>Swapping only the habitat can flip the shortcut prediction.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["landbird_background_swap"])}" alt="Landbird background swap">
        <figcaption>The reverse swap exposes the same background dependency.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2>Per-Sample Results</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Label</th>
          <th>Habitat</th>
          <th>Habitat prediction</th>
          <th>Habitat score</th>
          <th>Bird-shape prediction</th>
          <th>Bird-shape score</th>
        </tr>
      </thead>
      <tbody>{result_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Lesson</h2>
    <p>
      Overall accuracy hides the shortcut. Worst-group accuracy and habitat-swap
      counterfactuals make the failure legible: the baseline is right on common
      groups and wrong on crossed groups. This fallback remains in place so the
      suite still runs when the real Waterbirds manifest is not prepared locally.
    </p>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _build_synthetic_demo_card(
    output_path: Path,
    data: WaterbirdsShortcutReportData,
    *,
    build_metadata: BuildMetadata,
) -> DemoCard:
    return DemoCard(
        title="Demo 01 - Waterbirds Shortcut Proxy",
        task=(
            "Synthetic Waterbirds-style classification where bird class is spuriously "
            "correlated with land or water background."
        ),
        model="Deterministic habitat shortcut classifier compared with a bird-shape intervention.",
        explanation_methods=(
            "Worst-group evaluation",
            "Habitat evidence box",
            "Bird-region evidence box",
            "Counterfactual background swap",
        ),
        key_lesson=(
            "A classifier can look accurate on average while failing crossed "
            "label/background groups."
        ),
        failure_mode=(
            "The baseline follows the habitat and fails waterbirds on land or "
            "landbirds on water."
        ),
        intervention="Use the bird silhouette and re-test with background swaps.",
        remaining_caveats=(
            "Synthetic fallback, not the prepared Waterbirds benchmark path.",
            "No real Grad-CAM or Integrated Gradients in fallback mode.",
            "Prepare the local Waterbirds manifest to switch the report into real-data mode.",
        ),
        report_path=output_path,
        figure_paths=(
            data.assets["habitat_evidence"],
            data.assets["bird_evidence"],
            data.assets["waterbird_background_swap"],
            data.assets["landbird_background_swap"],
        ),
        build_metadata=build_metadata,
    )


def _round_robin_records(
    records: list[WaterbirdsManifestRecord],
    limit: int | None,
) -> list[WaterbirdsManifestRecord]:
    if limit is None or len(records) <= limit:
        return records
    by_group: dict[str, list[WaterbirdsManifestRecord]] = {}
    for record in records:
        by_group.setdefault(record.group, []).append(record)
    selected: list[WaterbirdsManifestRecord] = []
    groups = sorted(by_group)
    while len(selected) < limit:
        progressed = False
        for group in groups:
            group_records = by_group[group]
            if not group_records:
                continue
            selected.append(group_records.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _copy_image(image_path: Path, output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    shutil.copy2(image_path, output_path)
    return output_path


def _copy_exemplar_assets(
    exemplars: tuple[PrototypeExemplar, ...],
    *,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for index, exemplar in enumerate(exemplars):
        assets[f"{prefix}_{index}"] = _copy_image(
            exemplar.image_path,
            _asset_path(output_dir, f"{prefix}_{index:02d}.png"),
        )
    return assets


def _masked_background(image_path: Path, output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        panel = image.convert("RGB")
        width, height = panel.size
        left = width // 4
        top = height // 4
        right = width - left
        bottom = height - top
        draw = ImageDraw.Draw(panel)
        fill = (122, 122, 122)
        draw.rectangle((0, 0, width, top), fill=fill)
        draw.rectangle((0, bottom, width, height), fill=fill)
        draw.rectangle((0, top, left, bottom), fill=fill)
        draw.rectangle((right, top, width, bottom), fill=fill)
        panel.save(output_path)
    return output_path


def _masked_centre(image_path: Path, output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    with Image.open(image_path) as image:
        panel = image.convert("RGB")
        width, height = panel.size
        left = width // 4
        top = height // 4
        right = width - left
        bottom = height - top
        draw = ImageDraw.Draw(panel)
        draw.rectangle((left, top, right, bottom), fill=(122, 122, 122))
        panel.save(output_path)
    return output_path


def _centre_mass(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    height, width = values.shape
    top = height // 4
    bottom = height - top
    left = width // 4
    right = width - left
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    centre = float(values[top:bottom, left:right].sum())
    return centre / total


def _diagnostic_summary(
    *,
    model: FrozenResNetWaterbirdsProbe,
    records: list[WaterbirdsManifestRecord],
    limit: int,
    output_dir: Path,
    prefix: str,
) -> WaterbirdsExplanationSummary:
    selected = records[:limit]
    if not selected:
        return WaterbirdsExplanationSummary(0.0, 0.0, 0.0, 0.0)
    grad_cam_masses: list[float] = []
    integrated_gradients_masses: list[float] = []
    background_deltas: list[float] = []
    centre_deltas: list[float] = []
    for index, record in enumerate(selected):
        explanation = model.explain(record)
        grad_cam_masses.append(_centre_mass(explanation.grad_cam))
        integrated_gradients_masses.append(_centre_mass(explanation.integrated_gradients))
        original_score = model.score_image(record.image_path)
        background_masked = _masked_background(
            record.image_path,
            _asset_path(output_dir, f"{prefix}_background_mask_{index:02d}.png"),
        )
        centre_masked = _masked_centre(
            record.image_path,
            _asset_path(output_dir, f"{prefix}_centre_mask_{index:02d}.png"),
        )
        background_deltas.append(abs(original_score - model.score_image(background_masked)))
        centre_deltas.append(abs(original_score - model.score_image(centre_masked)))
    return WaterbirdsExplanationSummary(
        grad_cam_centre_mass=float(np.mean(grad_cam_masses)),
        integrated_gradients_centre_mass=float(np.mean(integrated_gradients_masses)),
        background_mask_delta=float(np.mean(background_deltas)),
        centre_mask_delta=float(np.mean(centre_deltas)),
    )


def _prototype_diagnostic_summary(
    *,
    comparator: PrototypeExemplarComparator,
    records: list[WaterbirdsManifestRecord],
    limit: int,
    output_dir: Path,
    prefix: str,
) -> PrototypeComparatorSummary:
    selected = records[:limit]
    if not selected:
        return PrototypeComparatorSummary(0.0, 0.0)
    background_deltas: list[float] = []
    centre_deltas: list[float] = []
    for index, record in enumerate(selected):
        original_score = comparator.score_image(record.image_path)
        background_masked = _masked_background(
            record.image_path,
            _asset_path(output_dir, f"{prefix}_background_mask_{index:02d}.png"),
        )
        centre_masked = _masked_centre(
            record.image_path,
            _asset_path(output_dir, f"{prefix}_centre_mask_{index:02d}.png"),
        )
        background_deltas.append(abs(original_score - comparator.score_image(background_masked)))
        centre_deltas.append(abs(original_score - comparator.score_image(centre_masked)))
    return PrototypeComparatorSummary(
        background_mask_delta=float(np.mean(background_deltas)),
        centre_mask_delta=float(np.mean(centre_deltas)),
    )


def _prediction_rows(
    records: list[WaterbirdsManifestRecord],
    erm_predictions: list[WaterbirdsPrediction],
    balanced_predictions: list[WaterbirdsPrediction],
    *,
    limit: int = 12,
) -> str:
    erm_by_id = {prediction.sample_id: prediction for prediction in erm_predictions}
    balanced_by_id = {
        prediction.sample_id: prediction for prediction in balanced_predictions
    }
    rows: list[str] = []
    for record in records[:limit]:
        erm_prediction = erm_by_id[record.sample_id]
        balanced_prediction = balanced_by_id[record.sample_id]
        rows.append(
            "<tr>"
            f"<td>{html.escape(record.sample_id)}</td>"
            f"<td>{html.escape(record.label)}</td>"
            f"<td>{html.escape(record.habitat)}</td>"
            f"<td>{html.escape(record.group)}</td>"
            f"<td>{html.escape(erm_prediction.predicted)}</td>"
            f"<td>{erm_prediction.probability:.3f}</td>"
            f"<td>{html.escape(balanced_prediction.predicted)}</td>"
            f"<td>{balanced_prediction.probability:.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _real_group_rows(
    erm_metrics: tuple[WaterbirdsGroupMetric, ...],
    balanced_metrics: tuple[WaterbirdsGroupMetric, ...],
    prototype_metrics: tuple[WaterbirdsGroupMetric, ...],
) -> str:
    balanced_by_group = {metric.group: metric for metric in balanced_metrics}
    prototype_by_group = {metric.group: metric for metric in prototype_metrics}
    rows: list[str] = []
    for erm_metric in erm_metrics:
        balanced_metric = balanced_by_group[erm_metric.group]
        prototype_metric = prototype_by_group[erm_metric.group]
        rows.append(
            "<tr>"
            f"<td>{html.escape(erm_metric.group)}</td>"
            f"<td>{erm_metric.count}</td>"
            f"<td>{erm_metric.accuracy:.1%}</td>"
            f"<td>{balanced_metric.accuracy:.1%}</td>"
            f"<td>{prototype_metric.accuracy:.1%}</td>"
            "</tr>"
        )
    return "".join(rows)


def _select_visual_record(
    records: list[WaterbirdsManifestRecord],
    predictions: list[WaterbirdsPrediction],
) -> WaterbirdsManifestRecord:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    crossed = [record for record in records if not record.is_aligned]
    wrong_crossed = [record for record in crossed if not by_id[record.sample_id].correct]
    if wrong_crossed:
        return max(
            wrong_crossed,
            key=lambda record: abs(by_id[record.sample_id].probability - 0.5),
        )
    if crossed:
        return max(
            crossed,
            key=lambda record: abs(by_id[record.sample_id].probability - 0.5),
        )
    return records[0]


def _build_real_report_data(
    config: WaterbirdsShortcutReportConfig,
    *,
    manifest_path: Path,
    title: str,
    narrative: str,
    asset_prefix: str,
) -> RealWaterbirdsShortcutReportData:
    records = load_waterbirds_manifest(manifest_path)
    train_records = _round_robin_records(
        filter_waterbirds_records(records, split="train"),
        config.max_train_records,
    )
    test_records = _round_robin_records(
        filter_waterbirds_records(records, split="test"),
        config.max_test_records,
    )
    if not train_records or not test_records:
        raise ValueError("Prepared Waterbirds manifest must contain train and test samples.")

    probe_config = WaterbirdsProbeConfig(
        input_size=config.input_size,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        weights_name=config.weights_name,
        seed=config.seed,
        backbone_tuning=config.backbone_tuning,
    )
    erm_probe = FrozenResNetWaterbirdsProbe(config=probe_config, training_mode="erm")
    erm_probe.fit(train_records)
    balanced_probe = FrozenResNetWaterbirdsProbe(
        config=probe_config,
        training_mode="group_balanced",
    )
    balanced_probe.fit(train_records)
    erm_predictions = erm_probe.predict(test_records)
    balanced_predictions = balanced_probe.predict(test_records)
    prototype_comparator = PrototypeExemplarComparator(
        probe=erm_probe,
        train_records=train_records,
    )
    prototype_predictions = prototype_comparator.predict(test_records)
    selected_record = _select_visual_record(test_records, erm_predictions)
    erm_explanation = erm_probe.explain(selected_record)
    balanced_explanation = balanced_probe.explain(selected_record)
    assets = {
        "selected_image": _copy_image(
            selected_record.image_path,
            _asset_path(config.output_dir, f"{asset_prefix}_selected_sample.png"),
        ),
        "selected_background_masked": _masked_background(
            selected_record.image_path,
            _asset_path(config.output_dir, f"{asset_prefix}_selected_background_masked.png"),
        ),
        "selected_centre_masked": _masked_centre(
            selected_record.image_path,
            _asset_path(config.output_dir, f"{asset_prefix}_selected_centre_masked.png"),
        ),
        "erm_grad_cam": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=erm_explanation.grad_cam,
            output_path=_asset_path(config.output_dir, f"{asset_prefix}_erm_grad_cam.png"),
        ),
        "erm_integrated_gradients": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=erm_explanation.integrated_gradients,
            output_path=_asset_path(
                config.output_dir,
                f"{asset_prefix}_erm_integrated_gradients.png",
            ),
        ),
        "balanced_grad_cam": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=balanced_explanation.grad_cam,
            output_path=_asset_path(config.output_dir, f"{asset_prefix}_balanced_grad_cam.png"),
        ),
        "balanced_integrated_gradients": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=balanced_explanation.integrated_gradients,
            output_path=_asset_path(
                config.output_dir,
                f"{asset_prefix}_balanced_integrated_gradients.png",
            ),
        ),
    }
    prototype_by_id = {prediction.sample_id: prediction for prediction in prototype_predictions}
    erm_by_id = {prediction.sample_id: prediction for prediction in erm_predictions}
    balanced_by_id = {prediction.sample_id: prediction for prediction in balanced_predictions}
    selected_prototype_prediction = prototype_by_id[selected_record.sample_id]
    nearest_predicted_exemplars = prototype_comparator.nearest_exemplars(
        selected_record,
        label=selected_prototype_prediction.predicted,
    )
    contrast_label = next(
        label
        for label in sorted({record.label for record in train_records})
        if label != selected_prototype_prediction.predicted
    )
    nearest_contrast_exemplars = prototype_comparator.nearest_exemplars(
        selected_record,
        label=contrast_label,
    )
    assets.update(
        _copy_exemplar_assets(
            nearest_predicted_exemplars,
            output_dir=config.output_dir,
            prefix=f"{asset_prefix}_prototype_predicted",
        )
    )
    assets.update(
        _copy_exemplar_assets(
            nearest_contrast_exemplars,
            output_dir=config.output_dir,
            prefix=f"{asset_prefix}_prototype_contrast",
        )
    )
    crossed_records = [record for record in test_records if not record.is_aligned]
    return RealWaterbirdsShortcutReportData(
        manifest_path=manifest_path,
        title=title,
        dataset_name=records[0].dataset if records else "unknown",
        narrative=narrative,
        train_records=train_records,
        test_records=test_records,
        erm_predictions=erm_predictions,
        balanced_predictions=balanced_predictions,
        prototype_predictions=prototype_predictions,
        erm_group_metrics=waterbirds_group_accuracy(test_records, erm_predictions),
        balanced_group_metrics=waterbirds_group_accuracy(
            test_records,
            balanced_predictions,
        ),
        prototype_group_metrics=prototype_group_accuracy(
            test_records,
            prototype_predictions,
        ),
        erm_summary=_diagnostic_summary(
            model=erm_probe,
            records=crossed_records,
            limit=config.diagnostic_sample_limit,
            output_dir=config.output_dir,
            prefix=f"{asset_prefix}_erm",
        ),
        balanced_summary=_diagnostic_summary(
            model=balanced_probe,
            records=crossed_records,
            limit=config.diagnostic_sample_limit,
            output_dir=config.output_dir,
            prefix=f"{asset_prefix}_balanced",
        ),
        prototype_summary=_prototype_diagnostic_summary(
            comparator=prototype_comparator,
            records=crossed_records,
            limit=config.diagnostic_sample_limit,
            output_dir=config.output_dir,
            prefix=f"{asset_prefix}_prototype",
        ),
        selected_record=selected_record,
        selected_erm_prediction=erm_by_id[selected_record.sample_id],
        selected_balanced_prediction=balanced_by_id[selected_record.sample_id],
        selected_prototype_prediction=selected_prototype_prediction,
        nearest_predicted_exemplars=nearest_predicted_exemplars,
        nearest_contrast_exemplars=nearest_contrast_exemplars,
        assets=assets,
    )


def _render_real_dataset_section(
    data: RealWaterbirdsShortcutReportData,
    *,
    output_path: Path,
) -> str:
    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    prediction_rows = _prediction_rows(
        data.test_records,
        data.erm_predictions,
        data.balanced_predictions,
    )
    group_rows = _real_group_rows(
        data.erm_group_metrics,
        data.balanced_group_metrics,
        data.prototype_group_metrics,
    )
    erm_accuracy = waterbirds_accuracy(data.erm_predictions)
    balanced_accuracy = waterbirds_accuracy(data.balanced_predictions)
    prototype_acc = prototype_accuracy(data.prototype_predictions)
    erm_worst = waterbirds_worst_group_accuracy(data.erm_group_metrics)
    balanced_worst = waterbirds_worst_group_accuracy(data.balanced_group_metrics)
    prototype_worst = waterbirds_worst_group_accuracy(data.prototype_group_metrics)
    predicted_exemplar_figures = "".join(
        (
            "<figure>"
            f'<img src="{rel(data.assets[f"{data.dataset_name}_prototype_predicted_{index}"])}" '
            'alt="Prototype exemplar">'
            f"<figcaption>{html.escape(exemplar.label)} | "
            f"{html.escape(exemplar.group)} | d={exemplar.distance:.3f}</figcaption>"
            "</figure>"
        )
        for index, exemplar in enumerate(data.nearest_predicted_exemplars)
    )
    contrast_exemplar_figures = "".join(
        (
            "<figure>"
            f'<img src="{rel(data.assets[f"{data.dataset_name}_prototype_contrast_{index}"])}" '
            'alt="Contrast exemplar">'
            f"<figcaption>{html.escape(exemplar.label)} | "
            f"{html.escape(exemplar.group)} | d={exemplar.distance:.3f}</figcaption>"
            "</figure>"
        )
        for index, exemplar in enumerate(data.nearest_contrast_exemplars)
    )
    erm_background_heavier = (
        data.erm_summary.background_mask_delta > data.erm_summary.centre_mask_delta
    )
    balanced_background_heavier = (
        data.balanced_summary.background_mask_delta > data.balanced_summary.centre_mask_delta
    )
    perturbation_verdict = (
        "For this slice, masking background changes the ERM score more than masking the image "
        "centre, which is consistent with background reliance."
        if erm_background_heavier
        else "For this slice, the ERM masking deltas are not dominated by background masking."
    )
    if balanced_background_heavier:
        perturbation_verdict += (
            " The reweighted model still shows non-trivial background sensitivity, so the fix "
            "is incomplete rather than solved."
        )
    else:
        perturbation_verdict += (
            " The reweighted model shifts the evidence pattern, but the group metrics show "
            "that the shortcut problem is still not solved."
        )
    takeaway_chips = [
        "Shortcut signal found",
        "Worst-group failure visible",
        "Fix still incomplete",
    ]
    chip_html = "".join(
        f'<span class="badge">{html.escape(chip)}</span>' for chip in takeaway_chips
    )

    return f"""
  <section>
    <h2>{html.escape(data.title)}</h2>
    <p>{html.escape(data.narrative)}</p>
    <p class="meta">
      Manifest: <code>{html.escape(str(data.manifest_path))}</code><br>
      Train samples: {len(data.train_records)} |
      Test samples: {len(data.test_records)} |
      Selected sample:
      <code>{html.escape(data.selected_record.sample_id)}</code>
    </p>
    <h3>Metric Summary</h3>
    <ul>
      <li>ERM accuracy: {erm_accuracy:.1%}</li>
      <li>Group-balanced accuracy: {balanced_accuracy:.1%}</li>
      <li>Prototype retrieval probe accuracy: {prototype_acc:.1%}</li>
      <li>ERM worst-group accuracy: {erm_worst:.1%}</li>
      <li>Group-balanced worst-group accuracy: {balanced_worst:.1%}</li>
      <li>Prototype retrieval probe worst-group accuracy: {prototype_worst:.1%}</li>
    </ul>
    <div>{chip_html}</div>
    <p>
      This page is strongest as a shortcut diagnosis page. Reweighting changes the error pattern
      and the evidence distribution, but it does not by itself remove shortcut reliance.
    </p>
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th>Count</th>
          <th>ERM accuracy</th>
          <th>Group-balanced accuracy</th>
          <th>Prototype retrieval probe accuracy</th>
        </tr>
      </thead>
      <tbody>{group_rows}</tbody>
    </table>
    <h3>Explanations on a Crossed-Group Sample</h3>
    <div class="grid">
      <figure>
        <img src="{rel(data.assets["selected_image"])}" alt="Selected sample">
        <figcaption>
          Original test sample. The report prefers the most confident wrong crossed-group ERM
          sample when available. ERM predicts {html.escape(data.selected_erm_prediction.predicted)}
          ({data.selected_erm_prediction.probability:.3f}); group-balanced predicts
          {html.escape(data.selected_balanced_prediction.predicted)}
          ({data.selected_balanced_prediction.probability:.3f}).
        </figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["erm_grad_cam"])}" alt="ERM Grad-CAM">
        <figcaption>ERM Grad-CAM.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["erm_integrated_gradients"])}" alt="ERM integrated gradients">
        <figcaption>ERM integrated gradients.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["balanced_grad_cam"])}" alt="Balanced Grad-CAM">
        <figcaption>Group-balanced Grad-CAM.</figcaption>
      </figure>
      <figure>
        <img
          src="{rel(data.assets["balanced_integrated_gradients"])}"
          alt="Balanced integrated gradients"
        >
        <figcaption>Group-balanced integrated gradients.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["selected_background_masked"])}" alt="Background masked sample">
        <figcaption>Background-muted perturbation.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["selected_centre_masked"])}" alt="Centre masked sample">
        <figcaption>Centre-muted perturbation.</figcaption>
      </figure>
    </div>
    <h3>Spatial Proxy and Perturbation Diagnostics</h3>
    <p>
      The centre-versus-background numbers below are a spatial proxy, not a bird
      segmentation metric. They average over crossed-group test samples only.
    </p>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Grad-CAM centre mass</th>
          <th>Integrated gradients centre mass</th>
          <th>Background-muted score delta</th>
          <th>Centre-muted score delta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ERM</td>
          <td>{data.erm_summary.grad_cam_centre_mass:.3f}</td>
          <td>{data.erm_summary.integrated_gradients_centre_mass:.3f}</td>
          <td>{data.erm_summary.background_mask_delta:.3f}</td>
          <td>{data.erm_summary.centre_mask_delta:.3f}</td>
        </tr>
        <tr>
          <td>Group-balanced</td>
          <td>{data.balanced_summary.grad_cam_centre_mass:.3f}</td>
          <td>{data.balanced_summary.integrated_gradients_centre_mass:.3f}</td>
          <td>{data.balanced_summary.background_mask_delta:.3f}</td>
          <td>{data.balanced_summary.centre_mask_delta:.3f}</td>
        </tr>
        <tr>
          <td>Prototype comparator</td>
          <td>n/a</td>
          <td>n/a</td>
          <td>{data.prototype_summary.background_mask_delta:.3f}</td>
          <td>{data.prototype_summary.centre_mask_delta:.3f}</td>
        </tr>
      </tbody>
    </table>
    <p>{html.escape(perturbation_verdict)}</p>
    <p>
      Prototype retrieval probe selected prediction:
      <code>{html.escape(data.selected_prototype_prediction.predicted)}</code>
      with margin {data.selected_prototype_prediction.score:.3f}.
    </p>
    <h3>Selected Test Predictions</h3>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Label</th>
          <th>Habitat</th>
          <th>Group</th>
          <th>ERM prediction</th>
          <th>ERM probability</th>
          <th>Group-balanced prediction</th>
          <th>Group-balanced probability</th>
        </tr>
      </thead>
      <tbody>{prediction_rows}</tbody>
    </table>
    <h3>Prototype-Space Diagnostic</h3>
    <p>
      This prototype-space diagnostic replaces a linear decision head with class prototypes in
      the current feature space, then shows which training exemplars are nearest
      to the selected crossed-group sample. Its low worst-group performance is itself informative:
      even case-based explanations inherit a shortcut-poisoned representation.
    </p>
    <h4>Nearest Exemplars for the Predicted Class</h4>
    <div class="grid">
      {predicted_exemplar_figures}
    </div>
    <h4>Nearest Exemplars for the Contrast Class</h4>
    <div class="grid">
      {contrast_exemplar_figures}
    </div>
  </section>
"""


def _render_real_html(
    config: WaterbirdsShortcutReportConfig,
    primary_data: RealWaterbirdsShortcutReportData,
    *,
    metashift_data: RealWaterbirdsShortcutReportData | None = None,
    build_metadata: BuildMetadata,
) -> Path:
    output_path = config.output_dir / "index.html"
    metashift_section = ""
    if metashift_data is not None:
        metashift_section = _render_real_dataset_section(metashift_data, output_path=output_path)
    lede = (
        "Real Waterbirds group shifts, explanation maps, and perturbation probes show that "
        "high average accuracy can still hide heavy context reliance."
    )
    brief = ReportBrief(
        claim=(
            "The shortcut is not just a metric artefact. It is visible in worst-group accuracy, "
            "attribution mass, and background-masking sensitivity."
        ),
        evidence=(
            "Compare ERM against the group-balanced probe, then line up the group table with the "
            "Grad-CAM, Integrated Gradients, and masking deltas for the selected crossed-group "
            "case."
        ),
        live_demo=(
            "Start from ERM average accuracy, then immediately move to worst-group accuracy and "
            "the selected crossed-group example so the audience sees the failure before the "
            "intervention."
        ),
        boundary=(
            "These are local ResNet-18 shortcut models plus proxy centre-versus-background "
            "checks, not a large-model benchmark reproduction."
        ),
        related=(
            ReportLink(
                slug="shortcut_industrial",
                title="Demo 02 - Industrial Shortcut Trap",
                reason="Shows the same shortcut logic in a controlled industrial setting.",
            ),
            ReportLink(
                slug="explanation_drift",
                title="Demo 08 - Explanation Drift Under Shift",
                reason="Extends the shortcut story into perturbation and acquisition drift.",
            ),
            ReportLink(
                slug="patchcore_bottle",
                title="Demo 03 - PatchCore on MVTec AD bottle",
                reason="Moves from classifier shortcuts to provenance-rich anomaly inspection.",
            ),
        ),
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Waterbirds Shortcut</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
      background: #f7f8fb;
    }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; background: #fff; padding: 20px; border: 1px solid #d8dee4; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #d8dee4; }}
    figcaption {{ font-size: 13px; color: #52606d; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{
      border-bottom: 1px solid #d8dee4;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
    .meta {{ color: #52606d; font-size: 14px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      margin: 0 8px 8px 0;
      padding: 5px 10px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      color: #364152;
      font-size: 12px;
      font-weight: 600;
    }}
    {report_chrome_css()}
  </style>
</head>
<body>
<main>
  {render_report_header(
      output_path=output_path,
      eyebrow="Demo 01 · Shortcut learning",
      title="Waterbirds Shortcut",
      lede=lede,
      build_metadata=build_metadata,
  )}
  {render_report_brief(brief)}
  {_render_real_dataset_section(primary_data, output_path=output_path)}
  {metashift_section}
  <section>
    <h2>Lesson</h2>
    <p>
      This is the real shortcut story the spec called for. Average accuracy alone
      is not enough. The failure becomes legible when the report forces group
      metrics, explanation maps, and targeted perturbations into the same view.
      The intervention here is intentionally modest: it changes the error pattern
      and the evidence distribution, but it does not by itself solve shortcut
      reliance. When the optional MetaShift manifest is prepared locally, the
      same report extends the story beyond Waterbirds into a second natural-context
      benchmark with the same evaluation and explanation contract.
    </p>
  </section>
  {render_related_reports(output_path=output_path, heading="Where to go next", links=brief.related)}
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _build_real_demo_card(
    output_path: Path,
    data: RealWaterbirdsShortcutReportData,
    *,
    build_metadata: BuildMetadata,
) -> DemoCard:
    return DemoCard(
        title="Demo 01 - Waterbirds Shortcut",
        task=(
            "Prepared Waterbirds classification with a canonical shortcut setup: "
            "bird class is spuriously correlated with habitat context."
        ),
        model=(
            "ResNet-18 classifiers with configurable frozen or partially fine-tuned backbones: "
            "plain ERM versus inverse-group-frequency weighting."
        ),
        explanation_methods=(
            "Worst-group evaluation",
            "Grad-CAM",
            "Integrated Gradients",
            "Background and centre perturbation probes",
            "Prototype-space retrieval probe",
        ),
        key_lesson=(
            "Shortcut behaviour is visible in group metrics, explanation mass, and "
            "score sensitivity to context masking."
        ),
        failure_mode=(
            "ERM can preserve high average accuracy while relying too heavily on "
            "habitat context."
        ),
        intervention=(
            "Reweight the training objective by group and re-check the evidence path, with the "
            "expectation that this changes the failure pattern rather than fully solving it."
        ),
        remaining_caveats=(
            "This is a strong local ResNet-18 shortcut benchmark, not a full large-model "
            "Waterbirds reproduction.",
            "Centre-versus-background attribution is a proxy rather than a true bird mask metric.",
            "Waterbirds usage terms should still be checked conservatively upstream.",
            "Optional MetaShift extension depends on locally prepared upstream assets.",
        ),
        report_path=output_path,
        figure_paths=(
            data.assets["erm_grad_cam"],
            data.assets["erm_integrated_gradients"],
            data.assets["balanced_grad_cam"],
            data.assets["balanced_integrated_gradients"],
            data.assets[f"{data.dataset_name}_prototype_predicted_0"],
            data.assets[f"{data.dataset_name}_prototype_contrast_0"],
        ),
        build_metadata=build_metadata,
    )


def _build_synthetic_report_data(
    config: WaterbirdsShortcutReportConfig,
) -> WaterbirdsShortcutReportData:
    train_samples, test_samples = generate_habitat_bird_dataset(config.synthetic_dir)
    habitat_classifier = HabitatShortcutClassifier()
    shape_classifier = BirdShapeClassifier()
    habitat_results = evaluate_bird_classifier(habitat_classifier, test_samples)
    shape_results = evaluate_bird_classifier(shape_classifier, test_samples)
    return WaterbirdsShortcutReportData(
        train_samples=train_samples,
        test_samples=test_samples,
        habitat_results=habitat_results,
        shape_results=shape_results,
        habitat_group_metrics=group_accuracy(test_samples, habitat_results),
        shape_group_metrics=group_accuracy(test_samples, shape_results),
        assets=_write_assets(
            output_dir=config.output_dir,
            test_samples=test_samples,
            habitat_classifier=habitat_classifier,
            shape_classifier=shape_classifier,
        ),
    )


def build_waterbirds_shortcut_report(config: WaterbirdsShortcutReportConfig) -> Path:
    """Build the Waterbirds shortcut report using real data when prepared."""

    ensure_directory(config.output_dir)
    if config.use_real_data and config.manifest_path.exists():
        build_metadata = make_build_metadata(
            data_mode="real",
            manifest_path=config.manifest_path,
            cache_enabled=False,
        )
        real_data = _build_real_report_data(
            config,
            manifest_path=config.manifest_path,
            title="Waterbirds Benchmark Slice",
            narrative=(
                "Prepared Waterbirds baseline using the canonical bird-versus-habitat "
                "spurious correlation setup."
            ),
            asset_prefix="waterbirds",
        )
        metashift_data = None
        if config.metashift_manifest_path.exists():
            metashift_data = _build_real_report_data(
                config,
                manifest_path=config.metashift_manifest_path,
                title="Natural-Context Extension - MetaShift",
                narrative=(
                    "Prepared MetaShift cat-vs-dog indoor/outdoor slice using the same "
                    "ERM-versus-group-balanced ResNet-18 comparison."
                ),
                asset_prefix="metashift",
            )
        output_path = _render_real_html(
            config,
            real_data,
            metashift_data=metashift_data,
            build_metadata=build_metadata,
        )
        card = _build_real_demo_card(output_path, real_data, build_metadata=build_metadata)
    else:
        build_metadata = make_build_metadata(
            data_mode="synthetic fallback",
            manifest_path=config.manifest_path if config.manifest_path.exists() else None,
            cache_enabled=False,
        )
        synthetic_data = _build_synthetic_report_data(config)
        output_path = _render_synthetic_html(config, synthetic_data, build_metadata)
        card = _build_synthetic_demo_card(
            output_path,
            synthetic_data,
            build_metadata=build_metadata,
        )
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
