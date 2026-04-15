"""Static report for prediction stability versus explanation drift."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.synthetic import generate_industrial_shortcut_dataset
from xai_demo_suite.explain.contracts import BoundingBox
from xai_demo_suite.explain.drift import DriftMeasurement, perturb_image
from xai_demo_suite.models.classification import (
    HybridShortcutClassifier,
    mask_region,
    predict_label,
)
from xai_demo_suite.reports.cards import (
    DemoCard,
    save_demo_card,
    save_demo_index_for_output_root,
)
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import draw_box_on_image


@dataclass(frozen=True, slots=True)
class ExplanationDriftReportConfig:
    """Configuration for the synthetic explanation drift report."""

    output_dir: Path = Path("outputs/explanation_drift")
    synthetic_dir: Path = Path("outputs/explanation_drift/synthetic")


@dataclass(frozen=True, slots=True)
class DriftExample:
    """Report data for one perturbation."""

    measurement: DriftMeasurement
    image_path: Path
    evidence_path: Path


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _asset_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assets" / name


def _build_drift_examples(
    config: ExplanationDriftReportConfig,
) -> tuple[Path, list[DriftExample]]:
    _, test_samples = generate_industrial_shortcut_dataset(config.synthetic_dir)
    baseline_sample = next(
        sample for sample in test_samples if sample.sample_id == "test_defect_clean"
    )
    classifier = HybridShortcutClassifier()
    baseline_score = classifier.predict_score(baseline_sample.image_path)
    baseline_prediction = predict_label(baseline_score)
    baseline_region = classifier.evidence_region_for(baseline_sample.image_path)
    baseline_evidence = draw_box_on_image(
        image_path=baseline_sample.image_path,
        box=baseline_region,
        output_path=_asset_path(config.output_dir, "baseline_evidence.png"),
        colour=(220, 20, 60),
    )

    perturbations = ("brightness_up", "contrast_down", "blur")
    examples: list[DriftExample] = []
    for perturbation_name in perturbations:
        perturbed_path = perturb_image(
            baseline_sample.image_path,
            _asset_path(config.output_dir, f"{perturbation_name}.png"),
            perturbation_name,
        )
        examples.append(
            _measure_perturbation(
                classifier=classifier,
                baseline_score=baseline_score,
                baseline_prediction=baseline_prediction,
                baseline_region=baseline_region,
                perturbation_name=perturbation_name,
                image_path=perturbed_path,
                output_dir=config.output_dir,
            )
        )

    stamp_removed = mask_region(
        baseline_sample.image_path,
        baseline_sample.stamp_region,
        _asset_path(config.output_dir, "stamp_faded.png"),
    )
    examples.append(
        _measure_perturbation(
            classifier=classifier,
            baseline_score=baseline_score,
            baseline_prediction=baseline_prediction,
            baseline_region=baseline_region,
            perturbation_name="stamp_faded",
            image_path=stamp_removed,
            output_dir=config.output_dir,
        )
    )
    return baseline_evidence, examples


def _measure_perturbation(
    *,
    classifier: HybridShortcutClassifier,
    baseline_score: float,
    baseline_prediction: str,
    baseline_region: BoundingBox,
    perturbation_name: str,
    image_path: Path,
    output_dir: Path,
) -> DriftExample:
    region = classifier.evidence_region_for(image_path)
    score = classifier.predict_score(image_path)
    evidence_path = draw_box_on_image(
        image_path=image_path,
        box=region,
        output_path=_asset_path(output_dir, f"{perturbation_name}_evidence.png"),
        colour=(40, 160, 80),
    )
    measurement = DriftMeasurement(
        perturbation_name=perturbation_name,
        baseline_score=baseline_score,
        perturbed_score=score,
        baseline_prediction=baseline_prediction,
        perturbed_prediction=predict_label(score),
        baseline_region=baseline_region,
        perturbed_region=region,
    )
    return DriftExample(
        measurement=measurement,
        image_path=image_path,
        evidence_path=evidence_path,
    )


def _render_rows(examples: list[DriftExample]) -> str:
    rows: list[str] = []
    for example in examples:
        measurement = example.measurement
        rows.append(
            "<tr>"
            f"<td>{html.escape(measurement.perturbation_name)}</td>"
            f"<td>{measurement.baseline_prediction}</td>"
            f"<td>{measurement.perturbed_prediction}</td>"
            f"<td>{'yes' if measurement.prediction_changed else 'no'}</td>"
            f"<td>{measurement.score_shift:.6f}</td>"
            f"<td>{measurement.explanation_shift:.1f} px</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_html(
    *,
    config: ExplanationDriftReportConfig,
    baseline_evidence: Path,
    examples: list[DriftExample],
    output_path: Path,
) -> None:
    rows = _render_rows(examples)

    def rel(path: Path) -> str:
        return html.escape(_relative(path, output_path.parent))

    figures = "\n".join(
        _render_drift_figure(example=example, output_path=output_path)
        for example in examples
    )
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
    }}
    main {{ max-width: 1080px; margin: 0 auto; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
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
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>Explanation Drift Under Shift</h1>
  <p>
    This synthetic report separates prediction drift from explanation drift.
    The hybrid classifier can keep the defect prediction while moving from
    shortcut-stamp evidence to object-shape evidence.
  </p>

  <section>
    <h2>Drift Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Perturbation</th>
          <th>Baseline prediction</th>
          <th>Perturbed prediction</th>
          <th>Prediction changed</th>
          <th>Score shift</th>
          <th>Explanation shift</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Evidence Panels</h2>
    <div class="grid">
      <figure>
        <img src="{rel(baseline_evidence)}" alt="Baseline evidence">
        <figcaption>Baseline evidence region before perturbation.</figcaption>
      </figure>
      {figures}
    </div>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")


def _render_drift_figure(example: DriftExample, output_path: Path) -> str:
    evidence_src = html.escape(_relative(example.evidence_path, output_path.parent))
    perturbation = html.escape(example.measurement.perturbation_name)
    prediction = html.escape(example.measurement.perturbed_prediction)
    return (
        f"""
      <figure>
        <img src="{evidence_src}" alt="{perturbation} evidence">
        <figcaption>
          {perturbation}: prediction {prediction},
          explanation shift {example.measurement.explanation_shift:.1f} px.
        </figcaption>
      </figure>
"""
    )


def _build_demo_card(
    output_path: Path,
    baseline_evidence: Path,
    examples: list[DriftExample],
) -> DemoCard:
    return DemoCard(
        title="Demo 08 - Explanation Drift Under Shift",
        task="Synthetic perturbations showing prediction shift and explanation shift separately.",
        model="Deterministic hybrid shortcut/shape classifier.",
        explanation_methods=(
            "Evidence-region tracking",
            "Score-shift measurement",
            "Prediction-change measurement",
            "Perturbation counter-tests",
        ),
        key_lesson=(
            "A prediction can remain stable while the evidence region moves, so "
            "explanation drift is a separate signal."
        ),
        failure_mode="A nuisance-driven classifier can move evidence under acquisition changes.",
        intervention="Track evidence-region movement alongside score and prediction movement.",
        remaining_caveats=(
            "Synthetic didactic perturbations only.",
            "No real corruption benchmark yet.",
            "No PatchCore drift comparison yet.",
        ),
        report_path=output_path,
        figure_paths=(
            baseline_evidence,
            *(example.evidence_path for example in examples[:3]),
        ),
    )


def build_explanation_drift_report(config: ExplanationDriftReportConfig) -> Path:
    """Build the synthetic explanation drift report."""

    ensure_directory(config.output_dir)
    baseline_evidence, examples = _build_drift_examples(config)
    output_path = config.output_dir / "index.html"
    _render_html(
        config=config,
        baseline_evidence=baseline_evidence,
        examples=examples,
        output_path=output_path,
    )
    card = _build_demo_card(output_path, baseline_evidence, examples)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
