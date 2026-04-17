from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from xai_demo_suite.reports.review_pack import ReviewPackConfig, build_review_pack


def _write_demo_card(
    output_root: Path,
    slug: str,
    title: str,
    *,
    build_metadata: dict[str, object] | None = None,
) -> None:
    report_dir = output_root / slug
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_path = report_dir / "figure.png"
    Image.new("RGB", (32, 32), (40, 80, 120)).save(figure_path)
    (report_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (output_root / "index.html").write_text("<html></html>", encoding="utf-8")
    (report_dir / "demo_card.json").write_text(
        json.dumps(
            {
                "title": title,
                "key_lesson": "lesson",
                "failure_mode": "failure",
                "intervention": "intervention",
                "report_path": f"{slug}/index.html",
                "figure_paths": [f"{slug}/figure.png"],
                "build_metadata": build_metadata,
            }
        ),
        encoding="utf-8",
    )


def test_review_pack_writes_html_with_cards(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    build_metadata = {
        "git_sha": "abcdef1",
        "built_at_utc": "2026-04-17 12:00:00 UTC",
        "data_mode": "real",
        "manifest_path": "data/processed/mvtec_ad/bottle/manifest.jsonl",
        "cache_status": "disabled",
    }
    _write_demo_card(
        output_root,
        "patchcore_bottle",
        "Demo 03 - PatchCore on MVTec AD bottle",
        build_metadata=build_metadata,
    )
    _write_demo_card(
        output_root,
        "patchcore_bottle_wrn50",
        "Demo 03 - PatchCore on MVTec AD bottle",
    )
    _write_demo_card(
        output_root,
        "patchcore_wrong_normal",
        "Demo 04 - PatchCore Learns the Wrong Normal",
        build_metadata=build_metadata,
    )
    _write_demo_card(
        output_root,
        "explanation_drift",
        "Demo 08 - Explanation Drift Under Shift",
        build_metadata=build_metadata,
    )
    _write_demo_card(
        output_root,
        "explanation_drift_neu_quick",
        "Demo 08 - Explanation Drift Under Shift",
    )

    output_path = build_review_pack(
        ReviewPackConfig(
            output_dir=output_root / "review_pack",
            output_root=output_root,
        )
    )

    html = output_path.read_text(encoding="utf-8")
    assert "XAI Demo Suite Review Pack" in html
    assert "Demo 03 - PatchCore on MVTec AD bottle" in html
    assert "ChatGPT Handoff" in html
    assert "Local Dataset Readiness" in html
    assert "Recommended Walkthrough" in html
    assert "Review guide" in html
    assert "Build Coherence" in html
    assert "Published Variants" in html
    assert "Demo 08 - Explanation Drift Under Shift" in html
    assert "explanation_drift_neu_quick" not in html
