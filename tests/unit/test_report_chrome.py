from __future__ import annotations

from pathlib import Path

from xai_demo_suite.reports.report_chrome import (
    ReportBrief,
    ReportLink,
    render_related_reports,
    render_report_brief,
    render_report_header,
)


def test_render_report_header_links_to_local_hub_and_demo_card(tmp_path: Path) -> None:
    output_path = tmp_path / "outputs" / "waterbirds_shortcut" / "index.html"

    html = render_report_header(
        output_path=output_path,
        eyebrow="Demo 01",
        title="Waterbirds Shortcut",
        lede="Shortcut summary.",
    )

    assert "../index.html" in html
    assert "demo_card.html" in html
    assert "Waterbirds Shortcut" in html


def test_render_report_brief_and_related_reports_include_expected_copy(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "outputs" / "patchcore_bottle" / "index.html"
    brief = ReportBrief(
        claim="Claim text.",
        evidence="Evidence text.",
        live_demo="Live demo text.",
        boundary="Boundary text.",
        related=(
            ReportLink(
                slug="patchcore_logic",
                title="Demo 07 - PatchCore Logical Anomaly Limits",
                reason="Rule-level follow-on.",
            ),
        ),
    )

    brief_html = render_report_brief(brief)
    related_html = render_related_reports(
        output_path=output_path,
        heading="Where to go next",
        links=brief.related,
    )

    assert "Demo Brief" in brief_html
    assert "Core claim" in brief_html
    assert "Boundary text." in brief_html
    assert "../patchcore_logic/index.html" in related_html
    assert "Rule-level follow-on." in related_html
