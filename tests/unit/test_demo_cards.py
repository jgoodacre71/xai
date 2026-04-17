from __future__ import annotations

import json
from pathlib import Path

from xai_demo_suite.reports.build_metadata import BuildMetadata
from xai_demo_suite.reports.cards import (
    DemoCard,
    save_demo_card,
    save_demo_index,
    save_demo_index_for_output_root,
)


def _card(tmp_path: Path) -> DemoCard:
    report_path = tmp_path / "outputs" / "patchcore_bottle" / "index.html"
    figure_path = tmp_path / "outputs" / "patchcore_bottle" / "assets" / "query_patch.png"
    report_path.parent.mkdir(parents=True)
    figure_path.parent.mkdir(parents=True)
    report_path.write_text("<html></html>", encoding="utf-8")
    figure_path.write_bytes(b"figure")
    return DemoCard(
        title="Demo 03 - PatchCore",
        task="Anomaly detection.",
        model="PatchCore-style memory bank.",
        explanation_methods=("anomaly map", "nearest normal patches"),
        key_lesson="Provenance makes the score inspectable.",
        failure_mode="Coarse baseline.",
        intervention="Replace top patch.",
        remaining_caveats=("Not causal proof.",),
        report_path=report_path,
        figure_paths=(figure_path,),
    )


def test_save_demo_card_writes_required_sections(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "patchcore_bottle"
    card = _card(tmp_path)

    json_path, html_path = save_demo_card(card, output_dir)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")
    assert data["task"] == "Anomaly detection."
    assert data["model"] == "PatchCore-style memory bank."
    assert data["explanation_methods"] == ["anomaly map", "nearest normal patches"]
    assert data["key_lesson"] == "Provenance makes the score inspectable."
    assert data["failure_mode"] == "Coarse baseline."
    assert data["intervention"] == "Replace top patch."
    assert data["remaining_caveats"] == ["Not causal proof."]
    assert "Open report" in html


def test_save_demo_index_links_report_and_card(tmp_path: Path) -> None:
    card = _card(tmp_path)
    save_demo_card(card, tmp_path / "outputs" / "patchcore_bottle")

    index_path = save_demo_index((card,), tmp_path / "outputs" / "index.html")

    html = index_path.read_text(encoding="utf-8")
    assert "Demo 03 - PatchCore" in html
    assert 'class="demo-tile"' in html
    assert 'class="thumb"' in html
    assert "patchcore_bottle/index.html" in html
    assert "patchcore_bottle/demo_card.html" in html


def test_save_demo_index_for_output_root_discovers_cards(tmp_path: Path) -> None:
    card = _card(tmp_path)
    save_demo_card(card, tmp_path / "outputs" / "patchcore_bottle")

    second_report = tmp_path / "outputs" / "patchcore_limits" / "index.html"
    second_figure = tmp_path / "outputs" / "patchcore_limits" / "assets" / "overview.png"
    second_report.parent.mkdir(parents=True)
    second_figure.parent.mkdir(parents=True)
    second_report.write_text("<html></html>", encoding="utf-8")
    second_figure.write_bytes(b"figure")
    save_demo_card(
        DemoCard(
            title="Demo 05 - Limits",
            task="PatchCore limits.",
            model="PatchCore-style memory bank.",
            explanation_methods=("anomaly map",),
            key_lesson="Novelty is not logic.",
            failure_mode="No symbolic count.",
            intervention="Add logic layer.",
            remaining_caveats=("Synthetic.",),
            report_path=second_report,
            figure_paths=(second_figure,),
        ),
        tmp_path / "outputs" / "patchcore_limits",
    )

    index_path = save_demo_index_for_output_root(tmp_path / "outputs")

    html = index_path.read_text(encoding="utf-8")
    assert "Demo 03 - PatchCore" in html
    assert "Demo 05 - Limits" in html
    assert "Static local entry point" in html
    assert "patchcore_bottle/index.html" in html
    assert "patchcore_limits/index.html" in html


def test_save_demo_index_shows_prepared_dataset_summary_and_demo_order(tmp_path: Path) -> None:
    waterbirds_manifest = (
        tmp_path
        / "data"
        / "processed"
        / "waterbirds"
        / "waterbird_complete95_forest2water2"
        / "manifest.jsonl"
    )
    waterbirds_manifest.parent.mkdir(parents=True, exist_ok=True)
    waterbirds_manifest.write_text("", encoding="utf-8")

    patchcore_card = _card(tmp_path)
    waterbirds_report = tmp_path / "outputs" / "waterbirds_shortcut" / "index.html"
    waterbirds_figure = tmp_path / "outputs" / "waterbirds_shortcut" / "assets" / "overview.png"
    waterbirds_report.parent.mkdir(parents=True, exist_ok=True)
    waterbirds_figure.parent.mkdir(parents=True, exist_ok=True)
    waterbirds_report.write_text("<html></html>", encoding="utf-8")
    waterbirds_figure.write_bytes(b"figure")
    waterbirds_card = DemoCard(
        title="Demo 01 - Waterbirds Shortcut",
        task="Shortcut learning.",
        model="Frozen ResNet probe.",
        explanation_methods=("Grad-CAM",),
        key_lesson="Background reliance is visible.",
        failure_mode="Spurious context.",
        intervention="Reweight groups.",
        remaining_caveats=("Local manifest required.",),
        report_path=waterbirds_report,
        figure_paths=(waterbirds_figure,),
    )

    index_path = save_demo_index(
        (patchcore_card, waterbirds_card),
        tmp_path / "outputs" / "index.html",
    )

    html = index_path.read_text(encoding="utf-8")
    assert "Reports generated: 2" in html
    assert "Waterbirds: prepared" in html
    assert html.index("Demo 01 - Waterbirds Shortcut") < html.index("Demo 03 - PatchCore")


def test_save_demo_index_shows_build_metadata_pills(tmp_path: Path) -> None:
    report_path = tmp_path / "outputs" / "patchcore_bottle" / "index.html"
    figure_path = tmp_path / "outputs" / "patchcore_bottle" / "assets" / "query_patch.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("<html></html>", encoding="utf-8")
    figure_path.write_bytes(b"figure")

    index_path = save_demo_index(
        (
            DemoCard(
                title="Demo 03 - PatchCore",
                task="Anomaly detection.",
                model="PatchCore-style memory bank.",
                explanation_methods=("anomaly map",),
                key_lesson="Provenance makes the score inspectable.",
                failure_mode="Coarse baseline.",
                intervention="Replace top patch.",
                remaining_caveats=("Not causal proof.",),
                report_path=report_path,
                figure_paths=(figure_path,),
                build_metadata=BuildMetadata(
                    git_sha="abcdef1",
                    built_at_utc="2026-04-17 12:00:00 UTC",
                    data_mode="real",
                    manifest_path=None,
                    cache_status="disabled",
                ),
            ),
        ),
        tmp_path / "outputs" / "index.html",
    )

    html = index_path.read_text(encoding="utf-8")
    assert "SHA abcdef1" in html
    assert ">real<" in html
