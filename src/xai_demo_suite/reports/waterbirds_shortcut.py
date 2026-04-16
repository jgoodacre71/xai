"""Static report for the Waterbirds shortcut demo."""

from __future__ import annotations

import html
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
    WaterbirdsGroupMetric,
    WaterbirdsPrediction,
    WaterbirdsProbeConfig,
    accuracy,
    evaluate_bird_classifier,
    group_accuracy,
    waterbirds_accuracy,
    waterbirds_group_accuracy,
    waterbirds_worst_group_accuracy,
    worst_group_accuracy,
)
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
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
class RealWaterbirdsShortcutReportData:
    """Computed data for the manifest-backed Waterbirds report."""

    manifest_path: Path
    train_records: list[WaterbirdsManifestRecord]
    test_records: list[WaterbirdsManifestRecord]
    erm_predictions: list[WaterbirdsPrediction]
    balanced_predictions: list[WaterbirdsPrediction]
    erm_group_metrics: tuple[WaterbirdsGroupMetric, ...]
    balanced_group_metrics: tuple[WaterbirdsGroupMetric, ...]
    erm_summary: WaterbirdsExplanationSummary
    balanced_summary: WaterbirdsExplanationSummary
    selected_record: WaterbirdsManifestRecord
    assets: dict[str, Path]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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


def _build_synthetic_demo_card(output_path: Path, data: WaterbirdsShortcutReportData) -> DemoCard:
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
) -> str:
    balanced_by_group = {metric.group: metric for metric in balanced_metrics}
    rows: list[str] = []
    for erm_metric in erm_metrics:
        balanced_metric = balanced_by_group[erm_metric.group]
        rows.append(
            "<tr>"
            f"<td>{html.escape(erm_metric.group)}</td>"
            f"<td>{erm_metric.count}</td>"
            f"<td>{erm_metric.accuracy:.1%}</td>"
            f"<td>{balanced_metric.accuracy:.1%}</td>"
            "</tr>"
        )
    return "".join(rows)


def _select_visual_record(
    records: list[WaterbirdsManifestRecord],
    predictions: list[WaterbirdsPrediction],
) -> WaterbirdsManifestRecord:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    crossed = [record for record in records if not record.is_aligned]
    for record in crossed:
        prediction = by_id[record.sample_id]
        if not prediction.correct:
            return record
    if crossed:
        return crossed[0]
    return records[0]


def _build_real_report_data(
    config: WaterbirdsShortcutReportConfig,
) -> RealWaterbirdsShortcutReportData:
    records = load_waterbirds_manifest(config.manifest_path)
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
    selected_record = _select_visual_record(test_records, erm_predictions)
    erm_explanation = erm_probe.explain(selected_record)
    balanced_explanation = balanced_probe.explain(selected_record)
    assets = {
        "selected_image": _copy_image(
            selected_record.image_path,
            _asset_path(config.output_dir, "selected_sample.png"),
        ),
        "selected_background_masked": _masked_background(
            selected_record.image_path,
            _asset_path(config.output_dir, "selected_background_masked.png"),
        ),
        "selected_centre_masked": _masked_centre(
            selected_record.image_path,
            _asset_path(config.output_dir, "selected_centre_masked.png"),
        ),
        "erm_grad_cam": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=erm_explanation.grad_cam,
            output_path=_asset_path(config.output_dir, "erm_grad_cam.png"),
        ),
        "erm_integrated_gradients": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=erm_explanation.integrated_gradients,
            output_path=_asset_path(config.output_dir, "erm_integrated_gradients.png"),
        ),
        "balanced_grad_cam": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=balanced_explanation.grad_cam,
            output_path=_asset_path(config.output_dir, "balanced_grad_cam.png"),
        ),
        "balanced_integrated_gradients": save_heatmap_overlay(
            image_path=selected_record.image_path,
            heatmap=balanced_explanation.integrated_gradients,
            output_path=_asset_path(config.output_dir, "balanced_integrated_gradients.png"),
        ),
    }
    crossed_records = [record for record in test_records if not record.is_aligned]
    return RealWaterbirdsShortcutReportData(
        manifest_path=config.manifest_path,
        train_records=train_records,
        test_records=test_records,
        erm_predictions=erm_predictions,
        balanced_predictions=balanced_predictions,
        erm_group_metrics=waterbirds_group_accuracy(test_records, erm_predictions),
        balanced_group_metrics=waterbirds_group_accuracy(
            test_records,
            balanced_predictions,
        ),
        erm_summary=_diagnostic_summary(
            model=erm_probe,
            records=crossed_records,
            limit=config.diagnostic_sample_limit,
            output_dir=config.output_dir,
            prefix="erm",
        ),
        balanced_summary=_diagnostic_summary(
            model=balanced_probe,
            records=crossed_records,
            limit=config.diagnostic_sample_limit,
            output_dir=config.output_dir,
            prefix="balanced",
        ),
        selected_record=selected_record,
        assets=assets,
    )


def _render_real_html(
    config: WaterbirdsShortcutReportConfig,
    data: RealWaterbirdsShortcutReportData,
) -> Path:
    output_path = config.output_dir / "index.html"

    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    prediction_rows = _prediction_rows(
        data.test_records,
        data.erm_predictions,
        data.balanced_predictions,
    )
    group_rows = _real_group_rows(data.erm_group_metrics, data.balanced_group_metrics)
    erm_accuracy = waterbirds_accuracy(data.erm_predictions)
    balanced_accuracy = waterbirds_accuracy(data.balanced_predictions)
    erm_worst = waterbirds_worst_group_accuracy(data.erm_group_metrics)
    balanced_worst = waterbirds_worst_group_accuracy(data.balanced_group_metrics)

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
    h1, h2 {{ margin: 0 0 12px; }}
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
  </style>
</head>
<body>
<main>
  <h1>Waterbirds Shortcut</h1>
  <p>
    This real-data Demo 01 path trains frozen ResNet-18 linear probes on the
    prepared Waterbirds manifest. The comparison is deliberately simple: plain
    ERM against inverse-group-frequency weighting. The point is to show that the
    shortcut is visible in group metrics, explanations, and perturbation tests.
  </p>
  <p class="meta">
    Manifest: <code>{html.escape(str(data.manifest_path))}</code><br>
    Train samples: {len(data.train_records)} |
    Test samples: {len(data.test_records)} |
    Selected sample:
    <code>{html.escape(data.selected_record.sample_id)}</code>
  </p>

  <section>
    <h2>Metric Summary</h2>
    <ul>
      <li>ERM accuracy: {erm_accuracy:.1%}</li>
      <li>Group-balanced accuracy: {balanced_accuracy:.1%}</li>
      <li>ERM worst-group accuracy: {erm_worst:.1%}</li>
      <li>Group-balanced worst-group accuracy: {balanced_worst:.1%}</li>
    </ul>
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th>Count</th>
          <th>ERM accuracy</th>
          <th>Group-balanced accuracy</th>
        </tr>
      </thead>
      <tbody>{group_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Explanations on a Crossed-Group Sample</h2>
    <div class="grid">
      <figure>
        <img src="{rel(data.assets["selected_image"])}" alt="Selected sample">
        <figcaption>
          Original test sample. The report prefers a crossed-group sample, and
          uses a misclassified ERM example when available.
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
  </section>

  <section>
    <h2>Spatial Proxy and Perturbation Diagnostics</h2>
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
      </tbody>
    </table>
  </section>

  <section>
    <h2>Selected Test Predictions</h2>
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
  </section>

  <section>
    <h2>Lesson</h2>
    <p>
      This is the real shortcut story the spec called for. Average accuracy alone
      is not enough. The failure becomes legible when the report forces group
      metrics, explanation maps, and targeted perturbations into the same view.
      The intervention here is intentionally modest, but it already shows how a
      different training objective can move the model away from background-heavy
      evidence.
    </p>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _build_real_demo_card(output_path: Path, data: RealWaterbirdsShortcutReportData) -> DemoCard:
    return DemoCard(
        title="Demo 01 - Waterbirds Shortcut",
        task=(
            "Prepared Waterbirds classification with a canonical shortcut setup: "
            "bird class is spuriously correlated with habitat context."
        ),
        model="Frozen ResNet-18 linear probes: plain ERM versus inverse-group-frequency weighting.",
        explanation_methods=(
            "Worst-group evaluation",
            "Grad-CAM",
            "Integrated Gradients",
            "Background and centre perturbation probes",
        ),
        key_lesson=(
            "Shortcut behaviour is visible in group metrics, explanation mass, and "
            "score sensitivity to context masking."
        ),
        failure_mode=(
            "ERM can preserve high average accuracy while relying too heavily on "
            "habitat context."
        ),
        intervention="Reweight the training objective by group and re-check the evidence path.",
        remaining_caveats=(
            "Linear probes on frozen features are a serious local baseline, "
            "not a full end-to-end benchmark reproduction.",
            "Centre-versus-background attribution is a proxy rather than a true bird mask metric.",
            "Waterbirds usage terms should still be checked conservatively upstream.",
        ),
        report_path=output_path,
        figure_paths=(
            data.assets["erm_grad_cam"],
            data.assets["erm_integrated_gradients"],
            data.assets["balanced_grad_cam"],
            data.assets["balanced_integrated_gradients"],
        ),
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
        real_data = _build_real_report_data(config)
        output_path = _render_real_html(config, real_data)
        card = _build_real_demo_card(output_path, real_data)
    else:
        synthetic_data = _build_synthetic_report_data(config)
        output_path = _render_synthetic_html(config, synthetic_data)
        card = _build_synthetic_demo_card(output_path, synthetic_data)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
