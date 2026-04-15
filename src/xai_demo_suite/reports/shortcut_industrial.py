"""Static report for the synthetic industrial shortcut lab."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.synthetic import (
    IndustrialShortcutSample,
    generate_industrial_shortcut_dataset,
)
from xai_demo_suite.models.classification.shortcut import (
    ClassificationResult,
    ShapeClassifier,
    StampShortcutClassifier,
    accuracy,
    evaluate_classifier,
    mask_region,
    swap_stamp,
)
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.vis.image_panels import draw_box_on_image


@dataclass(frozen=True, slots=True)
class IndustrialShortcutReportConfig:
    """Configuration for the synthetic industrial shortcut report."""

    output_dir: Path = Path("outputs/shortcut_industrial")
    synthetic_dir: Path = Path("outputs/shortcut_industrial/synthetic")


@dataclass(frozen=True, slots=True)
class ShortcutReportData:
    """Computed data for the shortcut report."""

    train_samples: list[IndustrialShortcutSample]
    test_samples: list[IndustrialShortcutSample]
    shortcut_results: list[ClassificationResult]
    shape_results: list[ClassificationResult]
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
    test_samples: list[IndustrialShortcutSample],
    shortcut_classifier: StampShortcutClassifier,
) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    swapped_normal = next(
        sample for sample in test_samples if sample.sample_id == "test_normal_swapped_stamp"
    )
    swapped_defect = next(
        sample for sample in test_samples if sample.sample_id == "test_defect_swapped_stamp"
    )

    assets["shortcut_evidence"] = draw_box_on_image(
        image_path=swapped_normal.image_path,
        box=swapped_normal.stamp_region,
        output_path=_asset_path(output_dir, "shortcut_evidence_stamp_box.png"),
    )
    assets["shape_evidence"] = draw_box_on_image(
        image_path=swapped_normal.image_path,
        box=swapped_normal.object_region,
        output_path=_asset_path(output_dir, "shape_evidence_object_box.png"),
        colour=(40, 160, 80),
    )
    masked_path = mask_region(
        swapped_normal.image_path,
        swapped_normal.stamp_region,
        _asset_path(output_dir, "counterfactual_stamp_removed.png"),
    )
    assets["stamp_removed"] = masked_path
    assets["stamp_swapped_defect"] = swap_stamp(
        swapped_defect.image_path,
        "red",
        _asset_path(output_dir, "counterfactual_defect_red_stamp.png"),
    )
    assets["score_drop_source"] = draw_box_on_image(
        image_path=masked_path,
        box=shortcut_classifier.stamp_region,
        output_path=_asset_path(output_dir, "counterfactual_stamp_removed_box.png"),
    )
    return assets


def _render_results_table(
    *,
    samples: list[IndustrialShortcutSample],
    shortcut_results: list[ClassificationResult],
    shape_results: list[ClassificationResult],
) -> str:
    shortcut_by_id = _result_map(shortcut_results)
    shape_by_id = _result_map(shape_results)
    rows: list[str] = []
    for sample in samples:
        shortcut = shortcut_by_id[sample.sample_id]
        shape = shape_by_id[sample.sample_id]
        rows.append(
            "<tr>"
            f"<td>{html.escape(sample.sample_id)}</td>"
            f"<td>{html.escape(sample.label)}</td>"
            f"<td>{html.escape(sample.shape)}</td>"
            f"<td>{html.escape(sample.stamp)}</td>"
            f"<td>{html.escape(shortcut.predicted)}</td>"
            f"<td>{shortcut.score:.3f}</td>"
            f"<td>{html.escape(shape.predicted)}</td>"
            f"<td>{shape.score:.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_html(config: IndustrialShortcutReportConfig, data: ShortcutReportData) -> Path:
    output_path = config.output_dir / "index.html"
    shortcut_accuracy = accuracy(data.shortcut_results)
    shape_accuracy = accuracy(data.shape_results)
    swapped = [
        sample
        for sample in data.test_samples
        if sample.sample_id in {"test_normal_swapped_stamp", "test_defect_swapped_stamp"}
    ]
    swapped_ids = {sample.sample_id for sample in swapped}
    swapped_shortcut_accuracy = accuracy(
        [result for result in data.shortcut_results if result.sample_id in swapped_ids]
    )
    swapped_shape_accuracy = accuracy(
        [result for result in data.shape_results if result.sample_id in swapped_ids]
    )
    rows = _render_results_table(
        samples=data.test_samples,
        shortcut_results=data.shortcut_results,
        shape_results=data.shape_results,
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
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>Industrial Shortcut Trap</h1>
  <p>
    This synthetic classification demo shows a model that appears successful
    when a corner stamp is correlated with the label, then fails when the stamp
    is swapped or removed.
  </p>

  <section>
    <h2>Metric Summary</h2>
    <ul>
      <li>Shortcut classifier accuracy: {shortcut_accuracy:.1%}</li>
      <li>Shape intervention accuracy: {shape_accuracy:.1%}</li>
      <li>Shortcut accuracy on swapped-stamp cases: {swapped_shortcut_accuracy:.1%}</li>
      <li>Shape accuracy on swapped-stamp cases: {swapped_shape_accuracy:.1%}</li>
    </ul>
  </section>

  <section>
    <h2>Evidence and Counterfactuals</h2>
    <div class="grid">
      <figure>
        <img src="{rel(data.assets["shortcut_evidence"])}" alt="Shortcut stamp evidence">
        <figcaption>The shortcut classifier looks at the corner stamp.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["shape_evidence"])}" alt="Shape evidence">
        <figcaption>The intervention looks at the central part silhouette.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["stamp_removed"])}" alt="Stamp removed counterfactual">
        <figcaption>Removing the stamp changes the shortcut evidence, not the part.</figcaption>
      </figure>
      <figure>
        <img src="{rel(data.assets["stamp_swapped_defect"])}" alt="Stamp swap counterfactual">
        <figcaption>Swapping the stamp can flip the shortcut model's decision.</figcaption>
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
          <th>Shape</th>
          <th>Stamp</th>
          <th>Shortcut prediction</th>
          <th>Shortcut score</th>
          <th>Shape prediction</th>
          <th>Shape score</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Lesson</h2>
    <p>
      The baseline succeeds for the wrong reason: the stamp is easier than the
      part. The intervention is not a final model; it is a controlled
      counter-test showing that removing the shortcut and using object evidence
      changes the failure mode.
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
        task="Synthetic classification where a corner stamp is spuriously correlated with defects.",
        model="Deterministic stamp shortcut classifier compared with a central-shape intervention.",
        explanation_methods=(
            "Region perturbation",
            "Counterfactual stamp removal",
            "Counterfactual stamp swap",
            "Object-region intervention",
        ),
        key_lesson=(
            "A classifier can appear accurate while using a nuisance stamp instead "
            "of the industrial part."
        ),
        failure_mode="The baseline follows the corner stamp and fails swapped-stamp cases.",
        intervention="Use the central part silhouette and counter-test with stamp swaps.",
        remaining_caveats=(
            "Synthetic didactic classifier, not a neural benchmark.",
            "No Grad-CAM or Integrated Gradients yet.",
            "No real industrial classification dataset yet.",
        ),
        report_path=output_path,
        figure_paths=(
            data.assets["shortcut_evidence"],
            data.assets["shape_evidence"],
            data.assets["stamp_removed"],
            data.assets["stamp_swapped_defect"],
        ),
    )


def build_industrial_shortcut_report(config: IndustrialShortcutReportConfig) -> Path:
    """Build the synthetic industrial shortcut report."""

    ensure_directory(config.output_dir)
    train_samples, test_samples = generate_industrial_shortcut_dataset(config.synthetic_dir)
    del train_samples
    shortcut_classifier = StampShortcutClassifier()
    shape_classifier = ShapeClassifier()
    shortcut_results = evaluate_classifier(shortcut_classifier, test_samples)
    shape_results = evaluate_classifier(shape_classifier, test_samples)
    assets = _write_assets(
        output_dir=config.output_dir,
        test_samples=test_samples,
        shortcut_classifier=shortcut_classifier,
    )
    data = ShortcutReportData(
        train_samples=[],
        test_samples=test_samples,
        shortcut_results=shortcut_results,
        shape_results=shape_results,
        assets=assets,
    )
    output_path = _render_html(config, data)
    card = _build_demo_card(output_path, data)
    save_demo_card(card, config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
