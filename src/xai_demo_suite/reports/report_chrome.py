"""Shared presentation helpers for static flagship reports."""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.reports.build_metadata import BuildMetadata


@dataclass(frozen=True, slots=True)
class ReportLink:
    """Link to a related report in the local suite."""

    slug: str
    title: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReportBrief:
    """Presenter-facing summary for a flagship report."""

    claim: str
    evidence: str
    live_demo: str
    boundary: str
    related: tuple[ReportLink, ...]


def report_chrome_css() -> str:
    """Return shared CSS for flagship report framing."""

    return """
    .hero {
      background: linear-gradient(180deg, #ffffff 0%, #f2f5fb 100%);
      border: 1px solid #d8dee4;
      padding: 24px;
    }
    .eyebrow {
      margin: 0 0 10px;
      color: #486581;
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .lede {
      max-width: 820px;
      color: #243b53;
      font-size: 17px;
      line-height: 1.55;
    }
    .action-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }
    .action-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 0 14px;
      border: 1px solid #bcccdc;
      color: #102a43;
      background: #ffffff;
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .summary-card {
      border: 1px solid #d8dee4;
      background: #ffffff;
      padding: 16px;
    }
    .summary-card h3 {
      margin: 0 0 10px;
      font-size: 15px;
      color: #102a43;
    }
    .summary-card p {
      margin: 0;
      color: #334e68;
      font-size: 14px;
      line-height: 1.5;
    }
    .related-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .related-link {
      display: block;
      border: 1px solid #d8dee4;
      background: #ffffff;
      padding: 16px;
      color: inherit;
      text-decoration: none;
    }
    .related-link strong {
      display: block;
      margin-bottom: 8px;
      color: #102a43;
      font-size: 15px;
    }
    .related-link span {
      color: #52606d;
      font-size: 14px;
      line-height: 1.45;
    }
    .build-meta {
      margin-top: 16px;
      padding: 14px;
      border: 1px solid #d8dee4;
      background: #ffffff;
    }
    .build-meta h3 {
      margin: 0 0 10px;
      font-size: 15px;
      color: #102a43;
    }
    .build-meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 10px;
    }
    .build-meta-grid div {
      font-size: 13px;
      line-height: 1.5;
      color: #334e68;
    }
    .build-meta-grid strong {
      display: block;
      color: #102a43;
    }
    """


def render_report_header(
    *,
    output_path: Path,
    eyebrow: str,
    title: str,
    lede: str,
    build_metadata: BuildMetadata | None = None,
) -> str:
    """Render a shared flagship report header."""

    local_hub = _relative_link(output_path, output_path.parent.parent / "index.html")
    demo_card = _relative_link(output_path, output_path.parent / "demo_card.html")
    metadata_html = ""
    if build_metadata is not None:
        metadata_html = render_build_metadata(build_metadata, output_path=output_path)
    return f"""
  <section class="hero">
    <p class="eyebrow">{html.escape(eyebrow)}</p>
    <h1>{html.escape(title)}</h1>
    <p class="lede">{html.escape(lede)}</p>
    <div class="action-row">
      <a class="action-link" href="{html.escape(local_hub)}">Open local demo hub</a>
      <a class="action-link" href="{html.escape(demo_card)}">Open demo card</a>
    </div>
    {metadata_html}
  </section>
"""


def render_build_metadata(build_metadata: BuildMetadata, *, output_path: Path) -> str:
    """Render build metadata inside the report header."""

    manifest_html = "Not used"
    if build_metadata.manifest_path is not None:
        manifest_html = html.escape(_relative_link(output_path, build_metadata.manifest_path))
    return f"""
    <div class="build-meta">
      <h3>Build Metadata</h3>
      <div class="build-meta-grid">
        <div><strong>Git SHA</strong>{html.escape(build_metadata.git_sha)}</div>
        <div><strong>Built at</strong>{html.escape(build_metadata.built_at_utc)}</div>
        <div><strong>Data mode</strong>{html.escape(build_metadata.data_mode)}</div>
        <div><strong>Cache</strong>{html.escape(build_metadata.cache_status)}</div>
        <div><strong>Manifest</strong><code>{manifest_html}</code></div>
      </div>
    </div>
"""


def render_report_brief(brief: ReportBrief) -> str:
    """Render presenter-facing summary cards."""

    cards = (
        ("Core claim", brief.claim),
        ("Best evidence on this page", brief.evidence),
        ("How to present it live", brief.live_demo),
        ("Boundary to say out loud", brief.boundary),
    )
    card_html = "".join(
        (
            "<div class=\"summary-card\">"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(text)}</p>"
            "</div>"
        )
        for title, text in cards
    )
    return f"""
  <section>
    <h2>Demo Brief</h2>
    <div class="summary-grid">
      {card_html}
    </div>
  </section>
"""


def render_related_reports(
    *,
    output_path: Path,
    heading: str,
    links: tuple[ReportLink, ...],
) -> str:
    """Render a small related-report section."""

    link_html_parts: list[str] = []
    for link in links:
        href = _relative_link(
            output_path,
            output_path.parent.parent / link.slug / "index.html",
        )
        link_html_parts.append(
            "<a class=\"related-link\" "
            f"href=\"{html.escape(href)}\">"
            f"<strong>{html.escape(link.title)}</strong>"
            f"<span>{html.escape(link.reason)}</span>"
            "</a>"
        )
    link_html = "".join(link_html_parts)
    return f"""
  <section>
    <h2>{html.escape(heading)}</h2>
    <div class="related-list">
      {link_html}
    </div>
  </section>
"""


def _relative_link(output_path: Path, target_path: Path) -> str:
    return Path(
        os.path.relpath(target_path.resolve(), start=output_path.parent.resolve())
    ).as_posix()
