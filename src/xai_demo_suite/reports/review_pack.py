"""Compact external-review pack for the local demo suite."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class ReviewPackConfig:
    """Configuration for the review pack."""

    output_dir: Path = Path("outputs/review_pack")
    output_root: Path = Path("outputs")
    docs_root: Path = Path("docs")


@dataclass(frozen=True, slots=True)
class ReviewCard:
    """Thin view of a generated demo card."""

    title: str
    key_lesson: str
    failure_mode: str
    intervention: str
    report_path: Path
    figure_paths: tuple[Path, ...]


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=root.resolve())).as_posix()


def _load_cards(output_root: Path) -> list[ReviewCard]:
    cards: list[ReviewCard] = []
    for card_path in sorted(output_root.glob("*/demo_card.json")):
        data = json.loads(card_path.read_text(encoding="utf-8"))
        cards.append(
            ReviewCard(
                title=str(data["title"]),
                key_lesson=str(data["key_lesson"]),
                failure_mode=str(data["failure_mode"]),
                intervention=str(data["intervention"]),
                report_path=output_root / str(data["report_path"]),
                figure_paths=tuple(
                    output_root / str(path) for path in data.get("figure_paths", [])
                ),
            )
        )
    return cards


def _dataset_readiness_items() -> tuple[tuple[str, str], ...]:
    return (
        (
            "Waterbirds",
            "data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl",
        ),
        (
            "MetaShift",
            "data/processed/metashift/subpopulation_shift_cat_dog_indoor_outdoor/manifest.jsonl",
        ),
        ("NEU-CLS", "data/processed/neu_cls/shortcut_binary/manifest.jsonl"),
        ("KolektorSDD2", "data/processed/ksdd2/shortcut_binary/manifest.jsonl"),
        ("MVTec AD bottle", "data/processed/mvtec_ad/bottle/manifest.jsonl"),
        ("MVTec LOCO AD", "data/processed/mvtec_loco_ad/juice_bottle/manifest.jsonl"),
        ("MVTec AD 2", "data/processed/mvtec_ad_2"),
        ("VisA", "data/processed/visa"),
    )


def _readiness_html(output_path: Path) -> str:
    items: list[str] = []
    for label, path_text in _dataset_readiness_items():
        path = Path(path_text)
        status = "prepared" if path.exists() else "not prepared"
        items.append(
            f"<li><strong>{html.escape(label)}</strong>: {status} "
            f"(<code>{html.escape(path.as_posix())}</code>)</li>"
        )
    return "".join(items)


def _cards_html(cards: list[ReviewCard], output_path: Path) -> str:
    sections: list[str] = []
    for card in cards:
        figure = ""
        if card.figure_paths:
            report_href = _relative(card.report_path, output_path.parent)
            figure = (
                f'<img src="{html.escape(_relative(card.figure_paths[0], output_path.parent))}" '
                f'alt="{html.escape(card.title)} preview">'
            )
        else:
            report_href = _relative(card.report_path, output_path.parent)
        sections.append(
            "<article class=\"card\">"
            f"<h3>{html.escape(card.title)}</h3>"
            f"<a class=\"report-link\" href=\"{html.escape(report_href)}\">Open report</a>"
            f"{figure}"
            f"<p><strong>Key lesson:</strong> {html.escape(card.key_lesson)}</p>"
            f"<p><strong>Failure mode:</strong> {html.escape(card.failure_mode)}</p>"
            f"<p><strong>Intervention:</strong> {html.escape(card.intervention)}</p>"
            "</article>"
        )
    return "".join(sections)


def _link(path: Path, output_path: Path, label: str) -> str:
    return f'<a href="{html.escape(_relative(path, output_path.parent))}">{html.escape(label)}</a>'


def build_review_pack(config: ReviewPackConfig) -> Path:
    """Write a compact HTML review pack for external reviewers and ChatGPT."""

    ensure_directory(config.output_dir)
    output_path = config.output_dir / "index.html"
    cards = _load_cards(config.output_root)
    hub_link = _link(config.output_root / "index.html", output_path, "the local demo hub")
    patchcore_link = _link(
        config.output_root / "patchcore_bottle" / "index.html",
        output_path,
        "Demo 03 - PatchCore hero report",
    )
    waterbirds_link = _link(
        config.output_root / "waterbirds_shortcut" / "index.html",
        output_path,
        "Demo 01 - Waterbirds shortcut report",
    )
    logic_link = _link(
        config.output_root / "patchcore_logic" / "index.html",
        output_path,
        "Demo 07 - LOCO logic limits",
    )
    drift_link = _link(
        config.output_root / "explanation_drift" / "index.html",
        output_path,
        "Demo 08 - Explanation drift",
    )
    caveat_items = "".join(
        [
            "<li>the repo is strong locally but still mixes benchmark-grade pieces "
            "with didactic report framing</li>",
            "<li>the industrial shortcut pillar is broader than before, but still "
            "not a full industrial benchmark family</li>",
            "<li>the PatchCore hero is strong and inspectable, but still not an "
            "official benchmark reproduction</li>",
            "<li>dataset licences vary; MVTec family is non-commercial, and local "
            "data artefacts are intentionally excluded from git</li>",
        ]
    )
    handoff_docs_item = (
        "<li>start with <code>README.md</code>, <code>REPO_SPEC.md</code>, "
        "<code>AGENTS.md</code>, <code>docs/DEMO_STATUS.md</code>, "
        "<code>docs/DEMO_CATALOGUE.md</code>, and <code>docs/DATASETS.md</code></li>"
    )
    handoff_reports_item = (
        "<li>then share screenshots or PDF exports of the flagship reports, "
        "because raw HTML is less convenient in normal chat</li>"
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XAI Demo Suite Review Pack</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
      background: #f7f8fb;
    }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    section {{ margin: 24px 0; background: #fff; padding: 20px; border: 1px solid #d8dee4; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .card {{
      border: 1px solid #d8dee4;
      padding: 16px;
      background: #fff;
    }}
    .card img {{
      width: 100%;
      height: auto;
      display: block;
      margin: 12px 0;
      border: 1px solid #d8dee4;
    }}
    .report-link {{
      display: inline-block;
      margin-bottom: 8px;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>XAI Demo Suite Review Pack</h1>
    <p>
      This pack is the shortest path for a reviewer to understand what is in the repo,
      which demos are strongest, which datasets are prepared locally on this machine,
      and what caveats still matter.
    </p>
    <p>
      Start with {hub_link},
      then the flagship reports for Demo 01, Demo 03, Demo 07, and Demo 08.
    </p>
  </section>

  <section>
    <h2>What This Repo Demonstrates</h2>
    <ul>
      <li>shortcut learning in natural and industrial settings</li>
      <li>PatchCore anomaly evidence with nearest-normal provenance</li>
      <li>where PatchCore fails on wrong normality, count, severity, and logic</li>
      <li>how prediction drift and explanation drift separate under nuisance shifts</li>
    </ul>
  </section>

  <section>
    <h2>Local Dataset Readiness</h2>
    <ul>{_readiness_html(output_path)}</ul>
  </section>

  <section>
    <h2>Best Entry Points</h2>
    <ul>
      <li>{patchcore_link}</li>
      <li>{waterbirds_link}</li>
      <li>{logic_link}</li>
      <li>{drift_link}</li>
    </ul>
  </section>

  <section>
    <h2>Demo Cards</h2>
    <div class="grid">{_cards_html(cards, output_path)}</div>
  </section>

  <section>
    <h2>Known Caveats</h2>
    <ul>{caveat_items}</ul>
  </section>

  <section>
    <h2>ChatGPT Handoff</h2>
    <p>For ChatGPT in chat, the cleanest order is:</p>
    <ol>
      <li>share the GitHub repo if available, or upload the repo docs first</li>
      {handoff_docs_item}
      {handoff_reports_item}
      <li>use this review pack plus the local demo hub as the navigation layer</li>
    </ol>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
