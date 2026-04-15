"""Static report for PatchCore novelty versus severity mismatch."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.synthetic import generate_severity_sweep_dataset
from xai_demo_suite.models.patchcore import ColourTexturePatchFeatureExtractor
from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index_for_output_root
from xai_demo_suite.reports.patchcore_synthetic_helpers import (
    SyntheticPatchCoreExample,
    build_or_load_synthetic_bank,
    build_synthetic_patchcore_examples,
    format_percentage,
    relative_path,
)
from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class PatchCoreSeverityReportConfig:
    """Configuration for the synthetic PatchCore severity report."""

    output_dir: Path = Path("outputs/patchcore_severity")
    cache_path: Path = Path("data/artefacts/patchcore/severity/severity_bank.npz")
    synthetic_dir: Path = Path("outputs/patchcore_severity/synthetic")
    patch_size: int = 80
    stride: int = 40
    top_k: int = 3
    use_cache: bool = True


def _severity_rank(examples: list[SyntheticPatchCoreExample]) -> list[str]:
    ranked = sorted(
        examples,
        key=lambda item: item.sample.severity_area,
        reverse=True,
    )
    return [example.sample.sample_id for example in ranked]


def _score_rank(examples: list[SyntheticPatchCoreExample]) -> list[str]:
    ranked = sorted(
        examples,
        key=lambda item: item.score.distance,
        reverse=True,
    )
    return [example.sample.sample_id for example in ranked]


def _render_rows(examples: list[SyntheticPatchCoreExample]) -> str:
    rows: list[str] = []
    for example in examples:
        rows.append(
            "<tr>"
            f"<td>{html.escape(example.sample.sample_id)}</td>"
            f"<td>{example.sample.severity_area}</td>"
            f"<td>{example.score.distance:.6f}</td>"
            f"<td>{format_percentage(example.mask_overlap.mask_covered_fraction)}</td>"
            f"<td>{html.escape(example.sample.semantic_note)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_example(example: SyntheticPatchCoreExample, output_path: Path) -> str:
    def rel(path: Path) -> str:
        return html.escape(relative_path(path, output_path.parent))

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
        <img src="{rel(example.assets["mask_overlay"])}" alt="Severity mask overlay">
        <figcaption>Synthetic severity proxy mask. Area: {example.sample.severity_area}</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["normal_source"])}" alt="Nearest normal source">
        <figcaption>Nearest nominal source patch used as provenance.</figcaption>
      </figure>
    </div>
  </section>
"""


def _render_html(
    *,
    config: PatchCoreSeverityReportConfig,
    examples: list[SyntheticPatchCoreExample],
    output_path: Path,
) -> None:
    rows = _render_rows(examples)
    sections = "\n".join(_render_example(example, output_path) for example in examples)
    severity_rank = " > ".join(html.escape(item) for item in _severity_rank(examples))
    score_rank = " > ".join(html.escape(item) for item in _score_rank(examples))
    ranks_match = _severity_rank(examples) == _score_rank(examples)
    mismatch_text = (
        "The order differs, so the largest synthetic severity proxy is not the highest "
        "PatchCore-style novelty case."
        if not ranks_match
        else "The order matches on this run; this still does not make the score calibrated."
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Severity Mismatch</title>
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
  <h1>PatchCore Severity Mismatch</h1>
  <p>
    Demo 06 isolates a common product mistake: treating a feature-space anomaly
    score as calibrated engineering severity. The synthetic masks provide a
    severity-area proxy; PatchCore-style scoring provides patch novelty.
  </p>

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>synthetic directory: <code>{html.escape(config.synthetic_dir.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(config.cache_path.as_posix())}</code></li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>colour_texture</code></li>
    </ul>
  </section>

  <section>
    <h2>Severity Versus Novelty</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Severity area proxy</th>
          <th>Top patch novelty score</th>
          <th>Mask covered</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <p>Severity rank: <code>{severity_rank}</code></p>
    <p>Novelty-score rank: <code>{score_rank}</code></p>
    <p>{html.escape(mismatch_text)}</p>
  </section>

{sections}

  <section>
    <h2>Lesson</h2>
    <p>
      PatchCore-style scores are useful triage signals, but they are not a
      calibrated severity model. Severity needs a separately defined target,
      measurement protocol, and validation against domain labels or physics.
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
        title="Demo 06 - PatchCore Severity Mismatch",
        task="Synthetic severity sweep comparing scratch area with PatchCore-style novelty score.",
        model="PatchCore-style nearest-neighbour memory bank with colour/texture patch features.",
        explanation_methods=(
            "Patch-score anomaly overlay",
            "Nearest-normal patch provenance",
            "Synthetic severity-mask comparison",
            "Severity-rank versus novelty-rank check",
        ),
        key_lesson=(
            "A high anomaly score is not automatically a high engineering severity."
        ),
        failure_mode=(
            "Feature-space novelty can rank a small high-contrast mark above a larger "
            "low-contrast defect."
        ),
        intervention=(
            "Add a calibrated severity target and validation layer instead of treating "
            "PatchCore distance as severity."
        ),
        remaining_caveats=(
            "Synthetic scratch sweep, not a production severity benchmark.",
            "No human-labelled severity data yet.",
            "No calibrated regression or ordinal severity model yet.",
        ),
        report_path=output_path,
        figure_paths=(
            examples[0].assets["score_overlay"],
            examples[1].assets["score_overlay"],
            examples[2].assets["mask_overlay"],
            examples[2].assets["normal_source"],
        ),
    )


def build_patchcore_severity_report(config: PatchCoreSeverityReportConfig) -> Path:
    """Build a static report for the synthetic PatchCore severity demo."""

    ensure_directory(config.output_dir)
    train_samples, eval_samples = generate_severity_sweep_dataset(config.synthetic_dir)
    extractor = ColourTexturePatchFeatureExtractor()
    memory_bank = build_or_load_synthetic_bank(
        train_samples=train_samples,
        extractor=extractor,
        cache_path=config.cache_path,
        use_cache=config.use_cache,
        patch_size=config.patch_size,
        stride=config.stride,
        category="slot_board_severity",
    )
    examples = build_synthetic_patchcore_examples(
        eval_samples=eval_samples,
        memory_bank=memory_bank,
        extractor=extractor,
        output_dir=config.output_dir,
        patch_size=config.patch_size,
        stride=config.stride,
        top_k=config.top_k,
    )
    output_path = config.output_dir / "index.html"
    _render_html(config=config, examples=examples, output_path=output_path)
    save_demo_card(_build_demo_card(output_path, examples), config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
