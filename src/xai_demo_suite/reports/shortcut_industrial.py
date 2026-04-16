"""Static report for the neural industrial shortcut demo."""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xai_demo_suite.data.industrial_manifest import (
    load_industrial_shortcut_manifest,
    manifest_records_to_samples,
)
from xai_demo_suite.data.synthetic import (
    IndustrialShortcutSample,
    generate_industrial_shortcut_dataset,
)
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.models.classification import (
    FrozenResNetIndustrialProbe,
    IndustrialPrediction,
    IndustrialProbeConfig,
    augment_stamp_invariant_samples,
    industrial_accuracy,
    mask_region,
)
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import save_heatmap_overlay


@dataclass(frozen=True, slots=True)
class IndustrialShortcutReportConfig:
    """Configuration for the industrial shortcut report."""

    output_dir: Path = Path("outputs/shortcut_industrial")
    synthetic_dir: Path = Path("outputs/shortcut_industrial/synthetic")
    input_size: int = 128
    batch_size: int = 16
    epochs: int = 18
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    weights_name: str | None = None
    seed: int = 13
    max_train_records: int | None = None
    diagnostic_sample_limit: int = 6
    real_manifest_path: Path = Path("data/processed/neu_cls/shortcut_binary/manifest.jsonl")
    use_real_data: bool = True


@dataclass(frozen=True, slots=True)
class IndustrialExplanationSummary:
    """Shortcut-versus-part explanation diagnostics."""

    grad_cam_stamp_mass: float
    grad_cam_object_mass: float
    integrated_stamp_mass: float
    integrated_object_mass: float
    stamp_mask_delta: float
    object_mask_delta: float


@dataclass(frozen=True, slots=True)
class ShortcutReportData:
    """Computed data for the industrial shortcut report."""

    train_samples: list[IndustrialShortcutSample]
    intervention_train_samples: list[IndustrialShortcutSample]
    test_samples: list[IndustrialShortcutSample]
    baseline_predictions: list[IndustrialPrediction]
    intervention_predictions: list[IndustrialPrediction]
    baseline_summary: IndustrialExplanationSummary
    intervention_summary: IndustrialExplanationSummary
    selected_sample: IndustrialShortcutSample
    assets: dict[str, Path]
    data_source_label: str


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=root.resolve())).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _challenge_samples(samples: list[IndustrialShortcutSample]) -> list[IndustrialShortcutSample]:
    return [sample for sample in samples if sample.variant != "clean"]


def _prediction_map(
    predictions: list[IndustrialPrediction],
) -> dict[str, IndustrialPrediction]:
    return {prediction.sample_id: prediction for prediction in predictions}


def _subset_accuracy(
    predictions: list[IndustrialPrediction],
    sample_ids: set[str],
) -> float:
    subset = [prediction for prediction in predictions if prediction.sample_id in sample_ids]
    return industrial_accuracy(subset)


def _object_mass(values: np.ndarray, sample: IndustrialShortcutSample) -> float:
    return _box_mass(values, sample.object_region)


def _stamp_mass(values: np.ndarray, sample: IndustrialShortcutSample) -> float:
    return _box_mass(values, sample.stamp_region)


def _box_mass(values: np.ndarray, box: BoundingBox) -> float:
    height, width = values.shape
    x = round(box.x * width / 128.0)
    y = round(box.y * height / 128.0)
    w = max(1, round(box.width * width / 128.0))
    h = max(1, round(box.height * height / 128.0))
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(values[y : y + h, x : x + w].sum()) / total


def _diagnostic_summary(
    *,
    model: FrozenResNetIndustrialProbe,
    samples: list[IndustrialShortcutSample],
    limit: int,
    output_dir: Path,
    prefix: str,
) -> IndustrialExplanationSummary:
    selected = samples[:limit]
    if not selected:
        return IndustrialExplanationSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    grad_cam_stamp: list[float] = []
    grad_cam_object: list[float] = []
    integrated_stamp: list[float] = []
    integrated_object: list[float] = []
    stamp_deltas: list[float] = []
    object_deltas: list[float] = []
    for sample in selected:
        explanation = model.explain(sample)
        grad_cam_stamp.append(_stamp_mass(explanation.grad_cam, sample))
        grad_cam_object.append(_object_mass(explanation.grad_cam, sample))
        integrated_stamp.append(_stamp_mass(explanation.integrated_gradients, sample))
        integrated_object.append(_object_mass(explanation.integrated_gradients, sample))

        baseline_score = model.score_image(sample.image_path)
        stamp_removed = mask_region(
            sample.image_path,
            sample.stamp_region,
            _asset_path(output_dir, f"{prefix}_{sample.sample_id}_stamp_removed.png"),
        )
        object_removed = mask_region(
            sample.image_path,
            sample.object_region,
            _asset_path(output_dir, f"{prefix}_{sample.sample_id}_object_removed.png"),
        )
        stamp_deltas.append(abs(baseline_score - model.score_image(stamp_removed)))
        object_deltas.append(abs(baseline_score - model.score_image(object_removed)))

    return IndustrialExplanationSummary(
        grad_cam_stamp_mass=float(np.mean(grad_cam_stamp)),
        grad_cam_object_mass=float(np.mean(grad_cam_object)),
        integrated_stamp_mass=float(np.mean(integrated_stamp)),
        integrated_object_mass=float(np.mean(integrated_object)),
        stamp_mask_delta=float(np.mean(stamp_deltas)),
        object_mask_delta=float(np.mean(object_deltas)),
    )


def _select_sample(
    samples: list[IndustrialShortcutSample],
    baseline_predictions: list[IndustrialPrediction],
) -> IndustrialShortcutSample:
    prediction_by_id = _prediction_map(baseline_predictions)
    for sample in _challenge_samples(samples):
        prediction = prediction_by_id[sample.sample_id]
        if not prediction.correct:
            return sample
    for sample in samples:
        if sample.variant == "swapped_stamp":
            return sample
    return samples[0]


def _write_assets(
    *,
    config: IndustrialShortcutReportConfig,
    selected_sample: IndustrialShortcutSample,
    baseline_model: FrozenResNetIndustrialProbe,
    intervention_model: FrozenResNetIndustrialProbe,
) -> dict[str, Path]:
    baseline_explanation = baseline_model.explain(selected_sample)
    intervention_explanation = intervention_model.explain(selected_sample)
    return {
        "selected_image": selected_sample.image_path,
        "baseline_grad_cam": save_heatmap_overlay(
            image_path=selected_sample.image_path,
            heatmap=baseline_explanation.grad_cam,
            output_path=_asset_path(config.output_dir, "baseline_grad_cam.png"),
        ),
        "baseline_integrated_gradients": save_heatmap_overlay(
            image_path=selected_sample.image_path,
            heatmap=baseline_explanation.integrated_gradients,
            output_path=_asset_path(config.output_dir, "baseline_integrated_gradients.png"),
        ),
        "intervention_grad_cam": save_heatmap_overlay(
            image_path=selected_sample.image_path,
            heatmap=intervention_explanation.grad_cam,
            output_path=_asset_path(config.output_dir, "intervention_grad_cam.png"),
        ),
        "intervention_integrated_gradients": save_heatmap_overlay(
            image_path=selected_sample.image_path,
            heatmap=intervention_explanation.integrated_gradients,
            output_path=_asset_path(config.output_dir, "intervention_integrated_gradients.png"),
        ),
        "stamp_removed": mask_region(
            selected_sample.image_path,
            selected_sample.stamp_region,
            _asset_path(config.output_dir, "selected_stamp_removed.png"),
        ),
        "object_removed": mask_region(
            selected_sample.image_path,
            selected_sample.object_region,
            _asset_path(config.output_dir, "selected_object_removed.png"),
        ),
    }


def _render_results_rows(
    samples: list[IndustrialShortcutSample],
    baseline_predictions: list[IndustrialPrediction],
    intervention_predictions: list[IndustrialPrediction],
) -> str:
    baseline_by_id = _prediction_map(baseline_predictions)
    intervention_by_id = _prediction_map(intervention_predictions)
    rows: list[str] = []
    for sample in samples:
        baseline_prediction = baseline_by_id[sample.sample_id]
        intervention_prediction = intervention_by_id[sample.sample_id]
        rows.append(
            "<tr>"
            f"<td>{html.escape(sample.sample_id)}</td>"
            f"<td>{html.escape(sample.label)}</td>"
            f"<td>{html.escape(sample.shape)}</td>"
            f"<td>{html.escape(sample.stamp)}</td>"
            f"<td>{html.escape(baseline_prediction.predicted)}</td>"
            f"<td>{baseline_prediction.probability:.3f}</td>"
            f"<td>{html.escape(intervention_prediction.predicted)}</td>"
            f"<td>{intervention_prediction.probability:.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_html(config: IndustrialShortcutReportConfig, data: ShortcutReportData) -> Path:
    output_path = config.output_dir / "index.html"
    test_ids = {sample.sample_id for sample in data.test_samples}
    swapped_ids = {
        sample.sample_id
        for sample in data.test_samples
        if sample.variant in {"swapped_stamp", "shifted_fixture"}
    }
    no_stamp_ids = {
        sample.sample_id
        for sample in data.test_samples
        if sample.variant == "no_stamp"
    }
    rows = _render_results_rows(
        data.test_samples,
        data.baseline_predictions,
        data.intervention_predictions,
    )
    baseline_accuracy = industrial_accuracy(data.baseline_predictions)
    intervention_accuracy = industrial_accuracy(data.intervention_predictions)
    baseline_swapped_accuracy = _subset_accuracy(data.baseline_predictions, swapped_ids)
    intervention_swapped_accuracy = _subset_accuracy(
        data.intervention_predictions,
        swapped_ids,
    )
    baseline_no_stamp_accuracy = _subset_accuracy(data.baseline_predictions, no_stamp_ids)
    intervention_no_stamp_accuracy = _subset_accuracy(
        data.intervention_predictions,
        no_stamp_ids,
    )

    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Industrial Shortcut Trap</title>
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
  <h1>Industrial Shortcut Trap</h1>
  <p>
    This upgraded Demo 02 uses {html.escape(data.data_source_label)} with learned
    probes. The baseline is trained on shortcut-correlated images. The
    intervention sees stamp-randomised and stamp-masked augmentations of the
    same parts.
  </p>

  <section>
    <h2>Metric Summary</h2>
    <ul>
      <li>Baseline overall accuracy: {baseline_accuracy:.1%}</li>
      <li>Intervention overall accuracy: {intervention_accuracy:.1%}</li>
      <li>Baseline swapped-fixture accuracy: {baseline_swapped_accuracy:.1%}</li>
      <li>Intervention swapped-fixture accuracy: {intervention_swapped_accuracy:.1%}</li>
      <li>Baseline no-stamp accuracy: {baseline_no_stamp_accuracy:.1%}</li>
      <li>Intervention no-stamp accuracy: {intervention_no_stamp_accuracy:.1%}</li>
      <li>Baseline train samples: {len(data.train_samples)}</li>
      <li>Intervention train samples: {len(data.intervention_train_samples)}</li>
      <li>Test samples: {len(test_ids)}</li>
    </ul>
  </section>

  <section>
    <h2>Explanations on a Shortcut Challenge Case</h2>
    <div class="grid">
      <figure>
        <img src="{rel(data.assets["selected_image"])}" alt="Selected challenge sample">
        <figcaption>Selected challenge sample for explanation comparison.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["baseline_grad_cam"])}" alt="Baseline Grad-CAM">
        <figcaption>Baseline Grad-CAM.</figcaption>
      </figure>
      <figure>
        <img
          src="{rel(data.assets["baseline_integrated_gradients"])}"
          alt="Baseline integrated gradients"
        >
        <figcaption>Baseline integrated gradients.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["intervention_grad_cam"])}" alt="Intervention Grad-CAM">
        <figcaption>Intervention Grad-CAM.</figcaption>
      </figure>
      <figure>
        <img
          src="{rel(data.assets["intervention_integrated_gradients"])}"
          alt="Intervention integrated gradients"
        >
        <figcaption>Intervention integrated gradients.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["stamp_removed"])}" alt="Stamp removed">
        <figcaption>Stamp-muted perturbation.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["object_removed"])}" alt="Object removed">
        <figcaption>Part-muted perturbation.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2>Shortcut Diagnostics on Challenge Cases</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Grad-CAM stamp mass</th>
          <th>Grad-CAM part mass</th>
          <th>IG stamp mass</th>
          <th>IG part mass</th>
          <th>Stamp-muted score delta</th>
          <th>Part-muted score delta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Baseline</td>
          <td>{data.baseline_summary.grad_cam_stamp_mass:.3f}</td>
          <td>{data.baseline_summary.grad_cam_object_mass:.3f}</td>
          <td>{data.baseline_summary.integrated_stamp_mass:.3f}</td>
          <td>{data.baseline_summary.integrated_object_mass:.3f}</td>
          <td>{data.baseline_summary.stamp_mask_delta:.3f}</td>
          <td>{data.baseline_summary.object_mask_delta:.3f}</td>
        </tr>
        <tr>
          <td>Intervention</td>
          <td>{data.intervention_summary.grad_cam_stamp_mass:.3f}</td>
          <td>{data.intervention_summary.grad_cam_object_mass:.3f}</td>
          <td>{data.intervention_summary.integrated_stamp_mass:.3f}</td>
          <td>{data.intervention_summary.integrated_object_mass:.3f}</td>
          <td>{data.intervention_summary.stamp_mask_delta:.3f}</td>
          <td>{data.intervention_summary.object_mask_delta:.3f}</td>
        </tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>Test Predictions</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Label</th>
          <th>Shape</th>
          <th>Stamp</th>
          <th>Baseline prediction</th>
          <th>Baseline probability</th>
          <th>Intervention prediction</th>
          <th>Intervention probability</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Lesson</h2>
    <p>
      The baseline can still look respectable on clean cases while tracking the
      fixture stamp too closely. The intervention is not magic; it simply removes
      the shortcut from the easiest path through training. That change shows up in
      both robustness numbers and explanation mass over the known stamp and part
      regions.
    </p>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _build_demo_card(output_path: Path, data: ShortcutReportData) -> DemoCard:
    return DemoCard(
        title="Demo 02 - Industrial Shortcut Trap",
        task=(
            "Industrial classification with a spuriously predictive fixture stamp, "
            "using a synthetic fallback or a prepared real-image NEU-CLS shortcut split."
        ),
        model=(
            "Compact convolutional probes over shortcut-correlated industrial images, "
            "with a baseline versus stamp-augmented intervention."
        ),
        explanation_methods=(
            "Grad-CAM",
            "Integrated Gradients",
            "Stamp and part perturbation probes",
            "Known-region attribution mass",
        ),
        key_lesson=(
            "A learned industrial model can still take the stamp shortcut if "
            "training allows it."
        ),
        failure_mode=(
            "The baseline overweights the fixture stamp and degrades on swapped "
            "or muted stamp cases."
        ),
        intervention=(
            "Train on stamp-randomised and stamp-masked augmentations of the "
            "same parts."
        ),
        remaining_caveats=(
            "The real-image path depends on a locally prepared NEU-CLS shortcut manifest.",
            "The real industrial shortcut still uses injected nuisance stamps to make "
            "the shortcut legible.",
            "The learned models are local demo baselines, not industrial benchmark systems.",
        ),
        report_path=output_path,
        figure_paths=(
            data.assets["baseline_grad_cam"],
            data.assets["baseline_integrated_gradients"],
            data.assets["intervention_grad_cam"],
            data.assets["intervention_integrated_gradients"],
        ),
    )


def build_industrial_shortcut_report(config: IndustrialShortcutReportConfig) -> Path:
    """Build the neural industrial shortcut report."""

    ensure_directory(config.output_dir)
    if config.use_real_data and config.real_manifest_path.exists():
        records = load_industrial_shortcut_manifest(config.real_manifest_path)
        samples = manifest_records_to_samples(records)
        train_samples = [sample for sample in samples if sample.split == "train"]
        test_samples = [sample for sample in samples if sample.split == "test"]
        data_source_label = "real NEU-CLS images with a prepared shortcut split"
    else:
        train_samples, test_samples = generate_industrial_shortcut_dataset(config.synthetic_dir)
        data_source_label = "synthetic industrial images"
    if config.max_train_records is not None:
        train_samples = train_samples[: config.max_train_records]

    intervention_train_samples = augment_stamp_invariant_samples(
        train_samples,
        output_dir=config.synthetic_dir / "intervention_train",
    )
    probe_config = IndustrialProbeConfig(
        input_size=config.input_size,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        weights_name=config.weights_name,
        seed=config.seed,
    )
    baseline_model = FrozenResNetIndustrialProbe(config=probe_config)
    baseline_model.fit(train_samples)
    intervention_model = FrozenResNetIndustrialProbe(config=probe_config)
    intervention_model.fit(intervention_train_samples)

    baseline_predictions = baseline_model.predict(test_samples)
    intervention_predictions = intervention_model.predict(test_samples)
    challenge_samples = _challenge_samples(test_samples)
    selected_sample = _select_sample(test_samples, baseline_predictions)
    assets = _write_assets(
        config=config,
        selected_sample=selected_sample,
        baseline_model=baseline_model,
        intervention_model=intervention_model,
    )
    data = ShortcutReportData(
        train_samples=train_samples,
        intervention_train_samples=intervention_train_samples,
        test_samples=test_samples,
        baseline_predictions=baseline_predictions,
        intervention_predictions=intervention_predictions,
        baseline_summary=_diagnostic_summary(
            model=baseline_model,
            samples=challenge_samples,
            limit=config.diagnostic_sample_limit,
            output_dir=config.output_dir,
            prefix="baseline",
        ),
        intervention_summary=_diagnostic_summary(
            model=intervention_model,
            samples=challenge_samples,
            limit=config.diagnostic_sample_limit,
            output_dir=config.output_dir,
            prefix="intervention",
        ),
        selected_sample=selected_sample,
        assets=assets,
        data_source_label=data_source_label,
    )
    output_path = _render_html(config, data)
    card = _build_demo_card(output_path, data)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
