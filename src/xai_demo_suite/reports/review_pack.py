"""Compact external-review pack for the local demo suite."""

from __future__ import annotations

import html
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.reports.build_metadata import BuildMetadata
from xai_demo_suite.reports.cards import KNOWN_PREPARED_DATASETS
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

    slug: str
    title: str
    key_lesson: str
    failure_mode: str
    intervention: str
    report_path: Path
    figure_paths: tuple[Path, ...]
    build_metadata: BuildMetadata | None
    demo_number: int | None


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=root.resolve())).as_posix()


def _load_cards(output_root: Path) -> list[ReviewCard]:
    cards: list[ReviewCard] = []
    for card_path in sorted(output_root.glob("*/demo_card.json")):
        data = json.loads(card_path.read_text(encoding="utf-8"))
        raw_build_metadata = data.get("build_metadata")
        build_metadata = None
        if isinstance(raw_build_metadata, dict):
            manifest_path = raw_build_metadata.get("manifest_path")
            build_metadata = BuildMetadata(
                git_sha=str(raw_build_metadata["git_sha"]),
                built_at_utc=str(raw_build_metadata["built_at_utc"]),
                data_mode=str(raw_build_metadata["data_mode"]),
                manifest_path=(
                    output_root / str(manifest_path) if isinstance(manifest_path, str) else None
                ),
                cache_status=str(raw_build_metadata["cache_status"]),
            )
        title = str(data["title"])
        cards.append(
            ReviewCard(
                slug=card_path.parent.name,
                title=title,
                key_lesson=str(data["key_lesson"]),
                failure_mode=str(data["failure_mode"]),
                intervention=str(data["intervention"]),
                report_path=output_root / str(data["report_path"]),
                figure_paths=tuple(
                    output_root / str(path) for path in data.get("figure_paths", [])
                ),
                build_metadata=build_metadata,
                demo_number=_demo_number(title),
            )
        )
    return cards


def _demo_number(title: str) -> int | None:
    match = re.search(r"Demo (\d+)", title)
    if match is None:
        return None
    return int(match.group(1))


def _readiness_status() -> tuple[tuple[str, bool, str], ...]:
    return tuple(
        (label, Path(path_text).exists(), path_text) for label, path_text in KNOWN_PREPARED_DATASETS
    )


def _readiness_html() -> str:
    items: list[str] = []
    for label, is_prepared, path_text in _readiness_status():
        status = "prepared" if is_prepared else "not prepared"
        items.append(
            f"<li><strong>{html.escape(label)}</strong>: {status} "
            f"(<code>{html.escape(path_text)}</code>)</li>"
        )
    return "".join(items)


def _preferred_demo_order() -> tuple[int, ...]:
    return (3, 4, 7, 2, 8, 1, 5, 6)


def _preferred_primary_slug(demo_number: int | None) -> str | None:
    if demo_number == 1:
        return "waterbirds_shortcut"
    if demo_number == 2:
        return "shortcut_industrial"
    if demo_number == 3:
        return "patchcore_bottle"
    if demo_number == 4:
        return "patchcore_wrong_normal"
    if demo_number == 5:
        return "patchcore_limits"
    if demo_number == 6:
        return "patchcore_severity"
    if demo_number == 7:
        return "patchcore_logic"
    if demo_number == 8:
        return "explanation_drift"
    return None


def _card_sort_key(card: ReviewCard) -> tuple[int, int, str]:
    preferred_slug = _preferred_primary_slug(card.demo_number)
    slug_rank = 0 if preferred_slug is not None and card.slug == preferred_slug else 1
    metadata_rank = 0 if card.build_metadata is not None else 1
    return (slug_rank, metadata_rank, card.slug)


def _curated_cards(cards: list[ReviewCard]) -> tuple[list[ReviewCard], list[ReviewCard]]:
    by_demo: dict[int, list[ReviewCard]] = defaultdict(list)
    uncategorised: list[ReviewCard] = []
    for card in cards:
        if card.demo_number is None:
            uncategorised.append(card)
            continue
        by_demo[card.demo_number].append(card)

    primary_cards: list[ReviewCard] = []
    variants: list[ReviewCard] = []
    for demo_number in _preferred_demo_order():
        demo_cards = by_demo.get(demo_number, [])
        if not demo_cards:
            continue
        ordered = sorted(demo_cards, key=_card_sort_key)
        primary = ordered[0]
        primary_cards.append(primary)
        for card in ordered[1:]:
            if card.title == primary.title and not card.slug.startswith("patchcore_bottle_"):
                continue
            variants.append(card)

    for card in uncategorised:
        primary_cards.append(card)

    variants.sort(key=lambda card: (card.demo_number or 999, card.title, card.slug))
    return primary_cards, variants


def _build_summary_html(cards: list[ReviewCard]) -> str:
    counts_by_sha: dict[str, int] = defaultdict(int)
    missing = 0
    for card in cards:
        if card.build_metadata is None:
            missing += 1
            continue
        counts_by_sha[card.build_metadata.git_sha] += 1

    if not counts_by_sha and missing == 0:
        return "<p>No build metadata was attached to the published cards.</p>"

    if len(counts_by_sha) == 1 and missing == 0:
        sha = next(iter(counts_by_sha))
        summary = (
            f"<p>All curated published cards share one build SHA: "
            f"<code>{html.escape(sha)}</code>.</p>"
        )
    else:
        summary = (
            "<p>The current published pack mixes build stamps. This is still reviewable, but a "
            "clean public release should ideally regenerate all curated outputs on one code SHA."
            "</p>"
        )

    rows = "".join(
        f"<li><code>{html.escape(sha)}</code>: {count} card(s)</li>"
        for sha, count in sorted(counts_by_sha.items())
    )
    missing_row = (
        f"<li>Missing build metadata: {missing} card(s)</li>" if missing else ""
    )
    return f"{summary}<ul>{rows}{missing_row}</ul>"


def _cards_html(cards: list[ReviewCard], output_path: Path) -> str:
    sections: list[str] = []
    for card in cards:
        report_href = _relative(card.report_path, output_path.parent)
        figure = ""
        if card.figure_paths:
            figure = (
                f'<img src="{html.escape(_relative(card.figure_paths[0], output_path.parent))}" '
                f'alt="{html.escape(card.title)} preview">'
            )
        metadata_html = ""
        if card.build_metadata is not None:
            metadata_html = (
                "<p><strong>Build:</strong> "
                f"<code>{html.escape(card.build_metadata.git_sha)}</code> · "
                f"{html.escape(card.build_metadata.data_mode)}</p>"
            )
        sections.append(
            "<article class=\"card\">"
            f"<h3>{html.escape(card.title)}</h3>"
            f"<a class=\"report-link\" href=\"{html.escape(report_href)}\">Open report</a>"
            f"{figure}"
            f"{metadata_html}"
            f"<p><strong>Key lesson:</strong> {html.escape(card.key_lesson)}</p>"
            f"<p><strong>Failure mode:</strong> {html.escape(card.failure_mode)}</p>"
            f"<p><strong>Intervention:</strong> {html.escape(card.intervention)}</p>"
            "</article>"
        )
    return "".join(sections)


def _link(path: Path, output_path: Path, label: str) -> str:
    return f'<a href="{html.escape(_relative(path, output_path.parent))}">{html.escape(label)}</a>'


def _doc_links(config: ReviewPackConfig, output_path: Path) -> str:
    items = [
        ("docs/REVIEW_GUIDE.md", "Review guide"),
        ("README.md", "README"),
        ("REPO_SPEC.md", "Repo spec"),
        ("AGENTS.md", "Agents"),
        ("docs/DEMO_STATUS.md", "Demo status"),
        ("docs/DEMO_CATALOGUE.md", "Demo catalogue"),
        ("docs/DATASETS.md", "Datasets"),
    ]
    links = [
        f"<li>{_link(Path(path_text), output_path, label)}</li>"
        for path_text, label in items
    ]
    return "".join(links)


def build_review_pack(config: ReviewPackConfig) -> Path:
    """Write a compact HTML review pack for external reviewers and ChatGPT."""

    ensure_directory(config.output_dir)
    output_path = config.output_dir / "index.html"
    all_cards = _load_cards(config.output_root)
    primary_cards, variant_cards = _curated_cards(all_cards)
    hub_link = _link(config.output_root / "index.html", output_path, "the local demo hub")
    patchcore_link = _link(
        config.output_root / "patchcore_bottle" / "index.html",
        output_path,
        "Demo 03 - PatchCore hero report",
    )
    wrong_normal_link = _link(
        config.output_root / "patchcore_wrong_normal" / "index.html",
        output_path,
        "Demo 04 - Wrong-normal report",
    )
    logic_link = _link(
        config.output_root / "patchcore_logic" / "index.html",
        output_path,
        "Demo 07 - LOCO logic limits",
    )
    industrial_link = _link(
        config.output_root / "shortcut_industrial" / "index.html",
        output_path,
        "Demo 02 - Industrial shortcut report",
    )
    drift_link = _link(
        config.output_root / "explanation_drift" / "index.html",
        output_path,
        "Demo 08 - Explanation drift",
    )
    waterbirds_link = _link(
        config.output_root / "waterbirds_shortcut" / "index.html",
        output_path,
        "Demo 01 - Waterbirds shortcut report",
    )
    review_guide_link = _link(config.docs_root / "REVIEW_GUIDE.md", output_path, "the review guide")
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
    walkthrough_items = "".join(
        [
            "<li>"
            f"Start at {patchcore_link} because it is the strongest single "
            "technical asset in the suite."
            "</li>",
            "<li>"
            f"Move to {wrong_normal_link} because it is the cleanest cautionary "
            "PatchCore story in the repo."
            "</li>",
            "<li>"
            f"Then open {logic_link} to show where local novelty is not enough "
            "for logic."
            "</li>",
            "<li>"
            f"Use {industrial_link} as the main classification shortcut page "
            "for an industrial audience."
            "</li>",
            "<li>"
            f"Finish with {drift_link} to separate prediction drift from "
            "explanation drift."
            "</li>",
            "<li>"
            f"Keep {waterbirds_link} as the natural-world shortcut appendix "
            "rather than the lead page."
            "</li>",
        ]
    )
    readiness = _readiness_status()
    prepared_count = sum(1 for _label, is_prepared, _path in readiness if is_prepared)
    variant_section = (
        f"""
  <section>
    <h2>Published Variants</h2>
    <div class="grid">{_cards_html(variant_cards, output_path)}</div>
  </section>
"""
        if variant_cards
        else ""
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
      Start with {hub_link}, use {review_guide_link} as the durable written companion,
      then move through Demo 03, Demo 04, Demo 07, Demo 02, Demo 08, and Demo 01.
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
    <p>Prepared dataset adapters detected: {prepared_count} / {len(readiness)}.</p>
    <ul>{_readiness_html()}</ul>
  </section>

  <section>
    <h2>Build Coherence</h2>
    {_build_summary_html(primary_cards + variant_cards)}
  </section>

  <section>
    <h2>Recommended Walkthrough</h2>
    <ol>{walkthrough_items}</ol>
  </section>

  <section>
    <h2>Best Entry Points</h2>
    <ul>
      <li>{patchcore_link}</li>
      <li>{wrong_normal_link}</li>
      <li>{logic_link}</li>
      <li>{industrial_link}</li>
      <li>{drift_link}</li>
      <li>{waterbirds_link}</li>
    </ul>
  </section>

  <section>
    <h2>Primary Published Reports</h2>
    <div class="grid">{_cards_html(primary_cards, output_path)}</div>
  </section>
{variant_section}

  <section>
    <h2>Known Caveats</h2>
    <ul>{caveat_items}</ul>
  </section>

  <section>
    <h2>Core Repo Docs</h2>
    <ul>{_doc_links(config, output_path)}</ul>
  </section>

  <section>
    <h2>ChatGPT Handoff</h2>
    <p>
      For ChatGPT review, the best path is the GitHub connector if your plan and
      experience expose it. The fallback is a ChatGPT Project with the repo docs
      and flagship report screenshots or PDF exports.
    </p>
    <ol>
      <li>
        share the GitHub repo if available, or create a Project and upload the
        repo docs first
      </li>
      {handoff_docs_item}
      {handoff_reports_item}
      <li>use this review pack plus the local demo hub as the navigation layer</li>
      <li>
        ask ChatGPT to assess spec coverage, demo quality, model honesty, and
        presentation clarity
      </li>
    </ol>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
