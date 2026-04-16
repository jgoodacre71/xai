"""Static report for learned prediction drift versus explanation drift."""

from __future__ import annotations

import html
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from xai_demo_suite.data.manifests import (
    ImageManifestRecord,
    filter_manifest_records,
    load_image_manifest,
)
from xai_demo_suite.data.synthetic import (
    IndustrialShortcutSample,
    generate_industrial_shortcut_dataset,
)
from xai_demo_suite.evaluate.localisation import PatchMaskOverlap, measure_patch_mask_overlap
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.explain.drift import DriftMeasurement, heatmap_drift, perturb_image
from xai_demo_suite.models.classification import (
    FrozenResNetIndustrialProbe,
    IndustrialPrediction,
    IndustrialProbeConfig,
    augment_stamp_invariant_samples,
    industrial_accuracy,
)
from xai_demo_suite.models.patchcore import (
    ColourTexturePatchFeatureExtractor,
    MeanRGBPatchFeatureExtractor,
    PatchCoreMemoryBank,
    PatchFeatureExtractor,
    build_patchcore_memory_bank,
    load_memory_bank,
    save_memory_bank,
    score_image_with_extractor,
)
from xai_demo_suite.models.patchcore.types import PatchScore
from xai_demo_suite.reports.cards import (
    DemoCard,
    save_demo_card,
    save_demo_index_for_output_root,
)
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import (
    save_heatmap_overlay,
    save_mask_overlay,
    save_score_overlay,
)

DEFAULT_DRIFT_CACHE_PATH = Path("data/artefacts/patchcore/bottle/drift_colour_texture_bank.npz")


@dataclass(frozen=True, slots=True)
class ExplanationDriftReportConfig:
    """Configuration for the drift report."""

    output_dir: Path = Path("outputs/explanation_drift")
    synthetic_dir: Path = Path("outputs/explanation_drift/synthetic")
    classifier_input_size: int = 128
    classifier_batch_size: int = 16
    classifier_epochs: int = 18
    classifier_seed: int = 19
    mvtec_manifest_path: Path = Path("data/processed/mvtec_ad/bottle/manifest.jsonl")
    mvtec_cache_path: Path = DEFAULT_DRIFT_CACHE_PATH
    mvtec_feature_extractor_name: str = "colour_texture"
    mvtec_max_train: int = 10
    mvtec_benchmark_limit: int = 16
    mvtec_patch_size: int = 128
    mvtec_stride: int = 128
    mvtec_top_k: int = 3
    include_mvtec_if_available: bool = True


@dataclass(frozen=True, slots=True)
class ClassifierPerturbationSummary:
    """Classifier drift summary for one perturbation."""

    perturbation_name: str
    accuracy: float
    prediction_changed: bool
    score_shift: float
    grad_cam_drift: float
    integrated_gradients_drift: float
    stamp_mass: float
    object_mass: float
    overlay_path: Path


@dataclass(frozen=True, slots=True)
class ClassifierDriftReport:
    """Learned classifier drift summary for one model."""

    label: str
    baseline_accuracy: float
    baseline_overlay_path: Path
    perturbations: tuple[ClassifierPerturbationSummary, ...]


@dataclass(frozen=True, slots=True)
class AnomalyPerturbationSummary:
    """PatchCore drift summary for one perturbation."""

    perturbation_name: str
    image_auc: float | None
    top_score_shift: float
    top_patch_shift: float
    mask_covered_fraction: float | None
    overlay_path: Path


@dataclass(frozen=True, slots=True)
class AnomalyDriftReport:
    """Optional anomaly-detector drift report."""

    baseline_overlay_path: Path
    baseline_mask_overlay_path: Path
    example_sample_id: str
    baseline_top_score: float
    baseline_mask_overlap: PatchMaskOverlap | None
    perturbations: tuple[AnomalyPerturbationSummary, ...]


@dataclass(frozen=True, slots=True)
class ExplanationDriftReportData:
    """Top-level report data."""

    baseline_classifier: ClassifierDriftReport
    intervention_classifier: ClassifierDriftReport
    anomaly_report: AnomalyDriftReport | None
    anomaly_note: str | None = None


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _classifier_perturbations() -> tuple[str, ...]:
    return (
        "brightness_up",
        "lighting_warm",
        "contrast_down",
        "blur",
        "jpeg_low_quality",
        "shadow_band",
    )


def _anomaly_perturbations() -> tuple[str, ...]:
    return (
        "brightness_up",
        "lighting_warm",
        "contrast_down",
        "blur",
        "jpeg_low_quality",
    )


def _prediction_map(
    predictions: list[IndustrialPrediction],
) -> dict[str, IndustrialPrediction]:
    return {prediction.sample_id: prediction for prediction in predictions}


def _perturb_classifier_samples(
    samples: list[IndustrialShortcutSample],
    *,
    perturbation_name: str,
    output_dir: Path,
) -> list[IndustrialShortcutSample]:
    perturbed: list[IndustrialShortcutSample] = []
    for sample in samples:
        output_path = output_dir / perturbation_name / f"{sample.sample_id}.png"
        perturbed.append(
            replace(
                sample,
                image_path=perturb_image(sample.image_path, output_path, perturbation_name),
            )
        )
    return perturbed


def _stamp_mass(sample: IndustrialShortcutSample, values: object) -> float:
    return _box_mass(sample.stamp_region, values)


def _object_mass(sample: IndustrialShortcutSample, values: object) -> float:
    return _box_mass(sample.object_region, values)


def _box_mass(box: BoundingBox, values: object) -> float:
    array = np.asarray(values)
    height, width = array.shape
    scale_x = width / 128.0
    scale_y = height / 128.0
    x = round(box.x * scale_x)
    y = round(box.y * scale_y)
    w = max(1, round(box.width * scale_x))
    h = max(1, round(box.height * scale_y))
    total = float(array.sum())
    if total <= 0.0:
        return 0.0
    return float(array[y : y + h, x : x + w].sum()) / total


def _train_classifier_models(
    config: ExplanationDriftReportConfig,
) -> tuple[
    list[IndustrialShortcutSample],
    list[IndustrialShortcutSample],
    list[IndustrialShortcutSample],
    FrozenResNetIndustrialProbe,
    FrozenResNetIndustrialProbe,
]:
    train_samples, test_samples = generate_industrial_shortcut_dataset(config.synthetic_dir)
    intervention_train_samples = augment_stamp_invariant_samples(
        train_samples,
        output_dir=config.synthetic_dir / "drift_intervention_train",
    )
    probe_config = IndustrialProbeConfig(
        input_size=config.classifier_input_size,
        batch_size=config.classifier_batch_size,
        epochs=config.classifier_epochs,
        weights_name=None,
        seed=config.classifier_seed,
    )
    baseline_model = FrozenResNetIndustrialProbe(config=probe_config)
    baseline_model.fit(train_samples)
    intervention_model = FrozenResNetIndustrialProbe(config=probe_config)
    intervention_model.fit(intervention_train_samples)
    return (
        train_samples,
        intervention_train_samples,
        test_samples,
        baseline_model,
        intervention_model,
    )


def _build_classifier_report(
    *,
    label: str,
    model: FrozenResNetIndustrialProbe,
    test_samples: list[IndustrialShortcutSample],
    output_dir: Path,
    perturbation_root: Path,
) -> ClassifierDriftReport:
    baseline_sample = next(
        sample for sample in test_samples if sample.sample_id == "test_defect_clean"
    )
    baseline_explanation = model.explain(baseline_sample)
    baseline_overlay_path = save_heatmap_overlay(
        image_path=baseline_sample.image_path,
        heatmap=baseline_explanation.grad_cam,
        output_path=_asset_path(output_dir, f"{label}_baseline_grad_cam.png"),
    )
    baseline_predictions = model.predict(test_samples)
    baseline_prediction = _prediction_map(baseline_predictions)[baseline_sample.sample_id]

    summaries: list[ClassifierPerturbationSummary] = []
    for perturbation_name in _classifier_perturbations():
        perturbed_samples = _perturb_classifier_samples(
            test_samples,
            perturbation_name=perturbation_name,
            output_dir=perturbation_root / label,
        )
        perturbed_predictions = model.predict(perturbed_samples)
        perturbed_sample = next(
            sample for sample in perturbed_samples if sample.sample_id == baseline_sample.sample_id
        )
        perturbed_prediction = _prediction_map(perturbed_predictions)[perturbed_sample.sample_id]
        perturbed_explanation = model.explain(perturbed_sample)
        overlay_path = save_heatmap_overlay(
            image_path=perturbed_sample.image_path,
            heatmap=perturbed_explanation.grad_cam,
            output_path=_asset_path(output_dir, f"{label}_{perturbation_name}_grad_cam.png"),
        )
        summaries.append(
            ClassifierPerturbationSummary(
                perturbation_name=perturbation_name,
                accuracy=industrial_accuracy(perturbed_predictions),
                prediction_changed=baseline_prediction.predicted != perturbed_prediction.predicted,
                score_shift=abs(perturbed_prediction.score - baseline_prediction.score),
                grad_cam_drift=heatmap_drift(
                    baseline_explanation.grad_cam,
                    perturbed_explanation.grad_cam,
                ),
                integrated_gradients_drift=heatmap_drift(
                    baseline_explanation.integrated_gradients,
                    perturbed_explanation.integrated_gradients,
                ),
                stamp_mass=_stamp_mass(perturbed_sample, perturbed_explanation.grad_cam),
                object_mass=_object_mass(perturbed_sample, perturbed_explanation.grad_cam),
                overlay_path=overlay_path,
            )
        )

    return ClassifierDriftReport(
        label=label,
        baseline_accuracy=industrial_accuracy(baseline_predictions),
        baseline_overlay_path=baseline_overlay_path,
        perturbations=tuple(summaries),
    )


def _build_patchcore_extractor(config: ExplanationDriftReportConfig) -> PatchFeatureExtractor:
    if config.mvtec_feature_extractor_name == "colour_texture":
        return ColourTexturePatchFeatureExtractor()
    if config.mvtec_feature_extractor_name == "mean_rgb":
        return MeanRGBPatchFeatureExtractor()
    raise ValueError("mvtec_feature_extractor_name must be colour_texture or mean_rgb.")


def _build_or_load_mvtec_bank(
    *,
    config: ExplanationDriftReportConfig,
    extractor: PatchFeatureExtractor,
) -> PatchCoreMemoryBank:
    train_records = filter_manifest_records(
        load_image_manifest(config.mvtec_manifest_path),
        split="train",
        defect_type="good",
    )[: config.mvtec_max_train]
    if not train_records:
        raise ValueError("No nominal training records found for MVTec bottle drift section.")
    if config.mvtec_cache_path.exists():
        memory_bank = load_memory_bank(config.mvtec_cache_path)
        if memory_bank.feature_name == extractor.feature_name:
            return memory_bank
    memory_bank = build_patchcore_memory_bank(
        train_records,
        extractor=extractor,
        patch_size=config.mvtec_patch_size,
        stride=config.mvtec_stride,
    )
    save_memory_bank(memory_bank, config.mvtec_cache_path)
    return memory_bank


def _top_patch_score(
    record: ImageManifestRecord,
    *,
    config: ExplanationDriftReportConfig,
    memory_bank: PatchCoreMemoryBank,
    extractor: PatchFeatureExtractor,
    image_path: Path | None = None,
) -> PatchScore:
    scores = score_image_with_extractor(
        sample_id=record.sample_id,
        image_path=image_path or record.image_path,
        memory_bank=memory_bank,
        extractor=extractor,
        patch_size=config.mvtec_patch_size,
        stride=config.mvtec_stride,
        top_k=config.mvtec_top_k,
    )
    return scores[0]


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
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
        tie_end = index + 1
        while tie_end < len(ranked) and ranked[tie_end][1] == ranked[index][1]:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for ranked_index in range(index, tie_end):
            ranks[ranked[ranked_index][0]] = average_rank
        index = tie_end

    positive_rank_sum = sum(rank for label, rank in zip(labels, ranks, strict=True) if label)
    return (
        positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)
    ) / (positive_count * negative_count)


def _benchmark_auc(
    *,
    config: ExplanationDriftReportConfig,
    records: list[ImageManifestRecord],
    memory_bank: PatchCoreMemoryBank,
    extractor: PatchFeatureExtractor,
    perturbation_name: str | None = None,
    output_dir: Path | None = None,
) -> float | None:
    labels: list[bool] = []
    scores: list[float] = []
    for record in records:
        image_path = record.image_path
        if perturbation_name is not None and output_dir is not None:
            image_path = perturb_image(
                record.image_path,
                output_dir / perturbation_name / f"{record.sample_id.replace('/', '_')}.png",
                perturbation_name,
            )
        top_score = _top_patch_score(
            record,
            config=config,
            memory_bank=memory_bank,
            extractor=extractor,
            image_path=image_path,
        )
        labels.append(record.is_anomalous)
        scores.append(top_score.distance)
    return _roc_auc(labels, scores)


def _build_anomaly_report(
    config: ExplanationDriftReportConfig,
) -> tuple[AnomalyDriftReport | None, str | None]:
    if not config.include_mvtec_if_available:
        return None, "MVTec anomaly drift section disabled by configuration."
    if not config.mvtec_manifest_path.exists():
        return None, "Prepare local MVTec bottle data to enable the anomaly-drift section."

    extractor = _build_patchcore_extractor(config)
    memory_bank = _build_or_load_mvtec_bank(config=config, extractor=extractor)
    all_test_records = filter_manifest_records(
        load_image_manifest(config.mvtec_manifest_path),
        split="test",
    )
    benchmark_records = all_test_records[: config.mvtec_benchmark_limit]
    example_record = next(record for record in all_test_records if record.is_anomalous)
    baseline_top = _top_patch_score(
        example_record,
        config=config,
        memory_bank=memory_bank,
        extractor=extractor,
    )
    baseline_overlay_path = save_score_overlay(
        image_path=example_record.image_path,
        scores=score_image_with_extractor(
            sample_id=example_record.sample_id,
            image_path=example_record.image_path,
            memory_bank=memory_bank,
            extractor=extractor,
            patch_size=config.mvtec_patch_size,
            stride=config.mvtec_stride,
            top_k=config.mvtec_top_k,
        ),
        output_path=_asset_path(config.output_dir, "mvtec_baseline_score_overlay.png"),
    )
    if example_record.mask_path is None:
        baseline_overlap = None
        baseline_mask_overlay_path = baseline_overlay_path
    else:
        baseline_overlap = measure_patch_mask_overlap(
            mask_path=example_record.mask_path,
            patch_box=baseline_top.query_box,
            image_path=example_record.image_path,
        )
        baseline_mask_overlay_path = save_mask_overlay(
            image_path=example_record.image_path,
            mask_path=example_record.mask_path,
            output_path=_asset_path(config.output_dir, "mvtec_baseline_mask_overlay.png"),
        )

    perturbations: list[AnomalyPerturbationSummary] = []
    auc_baseline = _benchmark_auc(
        config=config,
        records=benchmark_records,
        memory_bank=memory_bank,
        extractor=extractor,
    )
    del auc_baseline
    for perturbation_name in _anomaly_perturbations():
        perturbed_image_path = perturb_image(
            example_record.image_path,
            _asset_path(config.output_dir, f"mvtec_{perturbation_name}.png"),
            perturbation_name,
        )
        perturbed_scores = score_image_with_extractor(
            sample_id=example_record.sample_id,
            image_path=perturbed_image_path,
            memory_bank=memory_bank,
            extractor=extractor,
            patch_size=config.mvtec_patch_size,
            stride=config.mvtec_stride,
            top_k=config.mvtec_top_k,
        )
        top_score = perturbed_scores[0]
        drift_measurement = DriftMeasurement(
            perturbation_name=perturbation_name,
            baseline_score=baseline_top.distance,
            perturbed_score=top_score.distance,
            baseline_prediction="anomalous",
            perturbed_prediction="anomalous",
            baseline_region=baseline_top.query_box,
            perturbed_region=top_score.query_box,
        )
        overlap = (
            measure_patch_mask_overlap(
                mask_path=example_record.mask_path,
                patch_box=top_score.query_box,
                image_path=perturbed_image_path,
            )
            if example_record.mask_path is not None
            else None
        )
        perturbations.append(
            AnomalyPerturbationSummary(
                perturbation_name=perturbation_name,
                image_auc=_benchmark_auc(
                    config=config,
                    records=benchmark_records,
                    memory_bank=memory_bank,
                    extractor=extractor,
                    perturbation_name=perturbation_name,
                    output_dir=config.output_dir / "mvtec_benchmark",
                ),
                top_score_shift=drift_measurement.score_shift,
                top_patch_shift=drift_measurement.explanation_shift,
                mask_covered_fraction=(
                    overlap.mask_covered_fraction if overlap is not None else None
                ),
                overlay_path=save_score_overlay(
                    image_path=perturbed_image_path,
                    scores=perturbed_scores,
                    output_path=_asset_path(
                        config.output_dir,
                        f"mvtec_{perturbation_name}_score_overlay.png",
                    ),
                ),
            )
        )

    return (
        AnomalyDriftReport(
            baseline_overlay_path=baseline_overlay_path,
            baseline_mask_overlay_path=baseline_mask_overlay_path,
            example_sample_id=example_record.sample_id,
            baseline_top_score=baseline_top.distance,
            baseline_mask_overlap=baseline_overlap,
            perturbations=tuple(perturbations),
        ),
        None,
    )


def _render_classifier_rows(report: ClassifierDriftReport) -> str:
    rows: list[str] = []
    for summary in report.perturbations:
        rows.append(
            "<tr>"
            f"<td>{html.escape(summary.perturbation_name)}</td>"
            f"<td>{summary.accuracy:.1%}</td>"
            f"<td>{'yes' if summary.prediction_changed else 'no'}</td>"
            f"<td>{summary.score_shift:.3f}</td>"
            f"<td>{summary.grad_cam_drift:.3f}</td>"
            f"<td>{summary.integrated_gradients_drift:.3f}</td>"
            f"<td>{summary.stamp_mass:.3f}</td>"
            f"<td>{summary.object_mass:.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_classifier_figures(
    report: ClassifierDriftReport,
    *,
    output_path: Path,
) -> str:
    figures = [
        (
            html.escape(_relative(report.baseline_overlay_path, output_path.parent)),
            f"{report.label} baseline Grad-CAM",
            f"{report.label}: baseline Grad-CAM.",
        )
    ]
    for summary in report.perturbations:
        figures.append(
            (
                html.escape(_relative(summary.overlay_path, output_path.parent)),
                f"{report.label} {summary.perturbation_name}",
                (
                    f"{report.label}: {summary.perturbation_name}, "
                    f"Grad-CAM drift {summary.grad_cam_drift:.3f}."
                ),
            )
        )
    return "\n".join(
        (
            "      <figure>"
            f'<img src="{src}" alt="{alt}">'
            f"<figcaption>{caption}</figcaption>"
            "</figure>"
        )
        for src, alt, caption in figures
    )


def _render_anomaly_rows(report: AnomalyDriftReport) -> str:
    rows: list[str] = []
    for summary in report.perturbations:
        auc_text = "n/a" if summary.image_auc is None else f"{summary.image_auc:.3f}"
        mask_text = (
            "n/a"
            if summary.mask_covered_fraction is None
            else f"{summary.mask_covered_fraction:.3f}"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(summary.perturbation_name)}</td>"
            f"<td>{auc_text}</td>"
            f"<td>{summary.top_score_shift:.3f}</td>"
            f"<td>{summary.top_patch_shift:.1f} px</td>"
            f"<td>{mask_text}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_anomaly_figures(report: AnomalyDriftReport, *, output_path: Path) -> str:
    figures = [
        (
            html.escape(_relative(report.baseline_overlay_path, output_path.parent)),
            "PatchCore baseline overlay",
            "PatchCore baseline score overlay.",
        ),
        (
            html.escape(_relative(report.baseline_mask_overlay_path, output_path.parent)),
            "PatchCore baseline mask overlay",
            "Ground-truth mask overlay for the selected anomaly.",
        ),
    ]
    for summary in report.perturbations:
        figures.append(
            (
                html.escape(_relative(summary.overlay_path, output_path.parent)),
                f"PatchCore {summary.perturbation_name}",
                (
                    f"{summary.perturbation_name}: top-patch shift "
                    f"{summary.top_patch_shift:.1f} px."
                ),
            )
        )
    return "\n".join(
        (
            "      <figure>"
            f'<img src="{src}" alt="{alt}">'
            f"<figcaption>{caption}</figcaption>"
            "</figure>"
        )
        for src, alt, caption in figures
    )


def _render_html(
    *,
    config: ExplanationDriftReportConfig,
    data: ExplanationDriftReportData,
    output_path: Path,
) -> None:
    baseline_rows = _render_classifier_rows(data.baseline_classifier)
    intervention_rows = _render_classifier_rows(data.intervention_classifier)
    baseline_figures = _render_classifier_figures(
        data.baseline_classifier,
        output_path=output_path,
    )
    intervention_figures = _render_classifier_figures(
        data.intervention_classifier,
        output_path=output_path,
    )
    anomaly_section = ""
    if data.anomaly_report is not None:
        anomaly_rows = _render_anomaly_rows(data.anomaly_report)
        anomaly_figures = _render_anomaly_figures(data.anomaly_report, output_path=output_path)
        baseline_mask_text = (
            "n/a"
            if data.anomaly_report.baseline_mask_overlap is None
            else f"{data.anomaly_report.baseline_mask_overlap.mask_covered_fraction:.3f}"
        )
        anomaly_section = f"""
  <section>
    <h2>Anomaly Detector Drift</h2>
    <p>
      Optional local MVTec bottle section using PatchCore-style scoring over
      prepared data. Example sample:
      <code>{html.escape(data.anomaly_report.example_sample_id)}</code>.
    </p>
    <ul>
      <li>Baseline top patch score: {data.anomaly_report.baseline_top_score:.3f}</li>
      <li>Baseline mask covered by top patch: {baseline_mask_text}</li>
    </ul>
    <table>
      <thead>
        <tr>
          <th>Perturbation</th>
          <th>Image AUC</th>
          <th>Top-score shift</th>
          <th>Top-patch shift</th>
          <th>Mask covered by top patch</th>
        </tr>
      </thead>
      <tbody>{anomaly_rows}</tbody>
    </table>
    <div class="grid">
      {anomaly_figures}
    </div>
  </section>
"""
    elif data.anomaly_note is not None:
        anomaly_section = f"""
  <section>
    <h2>Anomaly Detector Drift</h2>
    <p>{html.escape(data.anomaly_note)}</p>
  </section>
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Explanation Drift Under Shift</title>
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
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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
  </style>
</head>
<body>
<main>
  <h1>Explanation Drift Under Shift</h1>
  <p>
    This upgraded Demo 08 separates prediction drift from explanation drift for
    a learned shortcut-sensitive classifier, and adds an optional local
    PatchCore anomaly-drift section when MVTec bottle data is prepared.
  </p>

  <section>
    <h2>Classifier Drift Summary</h2>
    <p>
      The baseline is trained on shortcut-correlated industrial images. The
      intervention is trained on stamp-randomised and stamp-masked variants of
      the same parts.
    </p>
    <h3>Baseline model</h3>
    <p>Baseline clean accuracy: {data.baseline_classifier.baseline_accuracy:.1%}</p>
    <table>
      <thead>
        <tr>
          <th>Perturbation</th>
          <th>Accuracy</th>
          <th>Prediction changed</th>
          <th>Score shift</th>
          <th>Grad-CAM drift</th>
          <th>IG drift</th>
          <th>Grad-CAM stamp mass</th>
          <th>Grad-CAM part mass</th>
        </tr>
      </thead>
      <tbody>{baseline_rows}</tbody>
    </table>
    <div class="grid">
      {baseline_figures}
    </div>

    <h3>Intervention model</h3>
    <p>Intervention clean accuracy: {data.intervention_classifier.baseline_accuracy:.1%}</p>
    <table>
      <thead>
        <tr>
          <th>Perturbation</th>
          <th>Accuracy</th>
          <th>Prediction changed</th>
          <th>Score shift</th>
          <th>Grad-CAM drift</th>
          <th>IG drift</th>
          <th>Grad-CAM stamp mass</th>
          <th>Grad-CAM part mass</th>
        </tr>
      </thead>
      <tbody>{intervention_rows}</tbody>
    </table>
    <div class="grid">
      {intervention_figures}
    </div>
  </section>
{anomaly_section}
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _build_demo_card(output_path: Path, data: ExplanationDriftReportData) -> DemoCard:
    figure_paths = [
        data.baseline_classifier.baseline_overlay_path,
        *(summary.overlay_path for summary in data.baseline_classifier.perturbations[:2]),
        data.intervention_classifier.baseline_overlay_path,
    ]
    if data.anomaly_report is not None:
        figure_paths.append(data.anomaly_report.baseline_overlay_path)
    return DemoCard(
        title="Demo 08 - Explanation Drift Under Shift",
        task=(
            "Corruption and acquisition-style shifts showing that performance "
            "drift and explanation drift are different signals."
        ),
        model=(
            "Learned industrial classifier drift path, plus optional local "
            "PatchCore anomaly-drift diagnostics on MVTec bottle."
        ),
        explanation_methods=(
            "Grad-CAM",
            "Integrated Gradients",
            "Known-region attribution mass",
            "PatchCore top-patch drift and mask checks",
        ),
        key_lesson=(
            "A metric can stay flat or move only slightly while the explanation "
            "path drifts into a nuisance region."
        ),
        failure_mode=(
            "Shortcut-sensitive models and anomaly detectors can move their "
            "attention under lighting, blur, compression, and acquisition shifts."
        ),
        intervention=(
            "Track explanation movement alongside score and accuracy, and compare "
            "against a shortcut-reduced classifier."
        ),
        remaining_caveats=(
            "The classifier path still uses synthetic industrial images.",
            "The anomaly section depends on local MVTec bottle preparation.",
            "Demo 07 still needs a logic-aware comparator.",
        ),
        report_path=output_path,
        figure_paths=tuple(figure_paths),
    )


def build_explanation_drift_report(config: ExplanationDriftReportConfig) -> Path:
    """Build the drift report."""

    ensure_directory(config.output_dir)
    (
        _train_samples,
        _intervention_train_samples,
        test_samples,
        baseline_model,
        intervention_model,
    ) = _train_classifier_models(config)
    baseline_classifier = _build_classifier_report(
        label="baseline",
        model=baseline_model,
        test_samples=test_samples,
        output_dir=config.output_dir,
        perturbation_root=config.output_dir / "classifier_perturbations",
    )
    intervention_classifier = _build_classifier_report(
        label="intervention",
        model=intervention_model,
        test_samples=test_samples,
        output_dir=config.output_dir,
        perturbation_root=config.output_dir / "classifier_perturbations",
    )
    anomaly_report, anomaly_note = _build_anomaly_report(config)
    data = ExplanationDriftReportData(
        baseline_classifier=baseline_classifier,
        intervention_classifier=intervention_classifier,
        anomaly_report=anomaly_report,
        anomaly_note=anomaly_note,
    )
    output_path = config.output_dir / "index.html"
    _render_html(config=config, data=data, output_path=output_path)
    card = _build_demo_card(output_path, data)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
