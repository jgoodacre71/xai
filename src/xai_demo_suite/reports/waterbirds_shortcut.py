"""Static report for a synthetic Waterbirds-style shortcut demo."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.synthetic import (
    HabitatBirdSample,
    generate_habitat_bird_dataset,
    write_habitat_counterfactual,
)
from xai_demo_suite.models.classification import (
    BirdShapeClassifier,
    ClassificationResult,
    GroupMetric,
    HabitatShortcutClassifier,
    accuracy,
    evaluate_bird_classifier,
    group_accuracy,
    worst_group_accuracy,
)
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import draw_box_on_image


@dataclass(frozen=True, slots=True)
class WaterbirdsShortcutReportConfig:
    """Configuration for the synthetic Waterbirds-style report."""

    output_dir: Path = Path("outputs/waterbirds_shortcut")
    synthetic_dir: Path = Path("outputs/waterbirds_shortcut/synthetic")


@dataclass(frozen=True, slots=True)
class WaterbirdsShortcutReportData:
    """Computed data for the Waterbirds-style shortcut report."""

    train_samples: list[HabitatBirdSample]
    test_samples: list[HabitatBirdSample]
    habitat_results: list[ClassificationResult]
    shape_results: list[ClassificationResult]
    habitat_group_metrics: tuple[GroupMetric, ...]
    shape_group_metrics: tuple[GroupMetric, ...]
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


def _render_html(
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
    This synthetic Demo 01 report mirrors the Waterbirds failure mode: a
    classifier can learn habitat background instead of bird evidence when
    training labels and habitats are strongly correlated.
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
      groups and wrong on crossed groups. The bird-shape rule is a controlled
      intervention, not a final classifier.
    </p>
  </section>
</main>
</body>
</html>
"""
    ensure_directory(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _build_demo_card(output_path: Path, data: WaterbirdsShortcutReportData) -> DemoCard:
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
            "Synthetic Waterbirds-style proxy, not the real Waterbirds benchmark.",
            "No neural classifier, Grad-CAM, or Integrated Gradients yet.",
            "No real dataset licence/citation path yet.",
        ),
        report_path=output_path,
        figure_paths=(
            data.assets["habitat_evidence"],
            data.assets["bird_evidence"],
            data.assets["waterbird_background_swap"],
            data.assets["landbird_background_swap"],
        ),
    )


def build_waterbirds_shortcut_report(config: WaterbirdsShortcutReportConfig) -> Path:
    """Build the synthetic Waterbirds-style shortcut report."""

    ensure_directory(config.output_dir)
    train_samples, test_samples = generate_habitat_bird_dataset(config.synthetic_dir)
    habitat_classifier = HabitatShortcutClassifier()
    shape_classifier = BirdShapeClassifier()
    habitat_results = evaluate_bird_classifier(habitat_classifier, test_samples)
    shape_results = evaluate_bird_classifier(shape_classifier, test_samples)
    data = WaterbirdsShortcutReportData(
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
    output_path = _render_html(config, data)
    card = _build_demo_card(output_path, data)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
