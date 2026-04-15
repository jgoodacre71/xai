"""Static report for PatchCore logical anomaly limitations."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.synthetic import generate_slot_board_dataset
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
class PatchCoreLogicReportConfig:
    """Configuration for the synthetic PatchCore logic report."""

    output_dir: Path = Path("outputs/patchcore_logic")
    cache_path: Path = Path("data/artefacts/patchcore/logic/logic_bank.npz")
    synthetic_dir: Path = Path("outputs/patchcore_logic/synthetic")
    patch_size: int = 80
    stride: int = 40
    top_k: int = 3
    use_cache: bool = True


def _logic_eval_examples(
    examples: list[SyntheticPatchCoreExample],
) -> list[SyntheticPatchCoreExample]:
    preferred = {"missing_one", "logic_swap"}
    return [example for example in examples if example.sample.sample_id in preferred]


def _render_rows(examples: list[SyntheticPatchCoreExample]) -> str:
    rows: list[str] = []
    for example in examples:
        native_statement = (
            "Top patch novelty and nearest normal patch"
            if example.sample.sample_id == "logic_swap"
            else "Top patch novelty around missing component"
        )
        required_statement = (
            "Two slots contain the wrong component identities"
            if example.sample.sample_id == "logic_swap"
            else "Slot 3 is empty"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(example.sample.sample_id)}</td>"
            f"<td>{html.escape(example.sample.label)}</td>"
            f"<td>{example.score.distance:.6f}</td>"
            f"<td>{format_percentage(example.mask_overlap.mask_covered_fraction)}</td>"
            f"<td>{html.escape(native_statement)}</td>"
            f"<td>{html.escape(required_statement)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_example(example: SyntheticPatchCoreExample, output_path: Path) -> str:
    def rel(path: Path) -> str:
        return html.escape(relative_path(path, output_path.parent))

    if example.sample.sample_id == "logic_swap":
        interpretation = (
            "PatchCore can highlight unusual local patches, but the claim that the "
            "red and blue components are in the wrong semantic positions is a rule "
            "outside the patch-distance model."
        )
    else:
        interpretation = (
            "A missing component is locally visible, but the statement that slot 3 "
            "is empty still comes from slot metadata or a component model."
        )
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
        <img src="{rel(example.assets["mask_overlay"])}" alt="Logic mask overlay">
        <figcaption>Synthetic mask for the changed local regions.</figcaption>
      </figure>
      <figure>
        <img src="{rel(example.assets["normal_source"])}" alt="Nearest normal source">
        <figcaption>Nearest nominal source patch used as provenance.</figcaption>
      </figure>
    </div>
    <p>{html.escape(interpretation)}</p>
  </section>
"""


def _render_html(
    *,
    config: PatchCoreLogicReportConfig,
    examples: list[SyntheticPatchCoreExample],
    output_path: Path,
) -> None:
    rows = _render_rows(examples)
    sections = "\n".join(_render_example(example, output_path) for example in examples)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PatchCore Logical Anomaly Limits</title>
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
    th, td {{
      border-bottom: 1px solid #d8dee4;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>PatchCore Logical Anomaly Limits</h1>
  <p>
    Demo 07 separates local novelty from logical understanding. The synthetic
    board gives PatchCore-style scoring a fair local anomaly signal, then asks
    for statements such as slot occupancy and component identity.
  </p>

  <section>
    <h2>Run Context</h2>
    <ul>
      <li>synthetic directory: <code>{html.escape(config.synthetic_dir.as_posix())}</code></li>
      <li>memory bank cache: <code>{html.escape(config.cache_path.as_posix())}</code></li>
      <li>patch size: {config.patch_size}, stride: {config.stride}, top-k: {config.top_k}</li>
      <li>feature extractor: <code>colour_texture</code></li>
      <li>MVTec LOCO AD comparison: not sourced yet.</li>
    </ul>
  </section>

  <section>
    <h2>Patch Novelty Versus Rule Statements</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Case</th>
          <th>Top patch score</th>
          <th>Mask covered</th>
          <th>Native PatchCore-style output</th>
          <th>Rule statement the product may need</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

{sections}

  <section>
    <h2>Lesson</h2>
    <p>
      PatchCore-style retrieval makes local evidence inspectable, but a logical
      anomaly product needs additional structure: slot definitions, component
      identity, expected relations, or a logic-aware model. This is why MVTec
      LOCO AD remains an important follow-up dataset.
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
        title="Demo 07 - PatchCore Logical Anomaly Limits",
        task="Synthetic slot-board comparison of local novelty and logical rule violations.",
        model="PatchCore-style nearest-neighbour memory bank with colour/texture patch features.",
        explanation_methods=(
            "Patch-score anomaly overlay",
            "Nearest-normal patch provenance",
            "Synthetic rule-statement comparison",
            "Structural versus logical case contrast",
        ),
        key_lesson=(
            "PatchCore can localise novelty, but it does not natively understand "
            "slot rules or component identities."
        ),
        failure_mode=(
            "The detector emits patch distances, not statements such as slot 3 is "
            "empty or two components are swapped."
        ),
        intervention=(
            "Add component detection, slot metadata, or logic-aware reasoning when "
            "the product needs rule-level explanations."
        ),
        remaining_caveats=(
            "Synthetic proxy, not an MVTec LOCO AD comparison yet.",
            "No symbolic or component-aware baseline yet.",
            "No real assembly-rule dataset yet.",
        ),
        report_path=output_path,
        figure_paths=(
            examples[0].assets["score_overlay"],
            examples[0].assets["normal_source"],
            examples[1].assets["score_overlay"],
            examples[1].assets["mask_overlay"],
        ),
    )


def build_patchcore_logic_report(config: PatchCoreLogicReportConfig) -> Path:
    """Build a static report for the synthetic PatchCore logical-anomaly demo."""

    ensure_directory(config.output_dir)
    train_samples, eval_samples = generate_slot_board_dataset(config.synthetic_dir)
    extractor = ColourTexturePatchFeatureExtractor()
    memory_bank = build_or_load_synthetic_bank(
        train_samples=train_samples,
        extractor=extractor,
        cache_path=config.cache_path,
        use_cache=config.use_cache,
        patch_size=config.patch_size,
        stride=config.stride,
        category="slot_board_logic",
    )
    all_examples = build_synthetic_patchcore_examples(
        eval_samples=eval_samples,
        memory_bank=memory_bank,
        extractor=extractor,
        output_dir=config.output_dir,
        patch_size=config.patch_size,
        stride=config.stride,
        top_k=config.top_k,
    )
    examples = _logic_eval_examples(all_examples)
    if len(examples) != 2:
        raise ValueError("Logic report requires missing_one and logic_swap examples.")
    output_path = config.output_dir / "index.html"
    _render_html(config=config, examples=examples, output_path=output_path)
    save_demo_card(_build_demo_card(output_path, examples), config.output_dir)
    save_demo_index_for_output_root(config.output_dir.parent)
    return output_path
