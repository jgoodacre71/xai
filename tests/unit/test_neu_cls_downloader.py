from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.neu_cls import (
    NEU_CLS_DATASET,
    NEU_CLS_LICENCE,
    build_neu_cls_shortcut_manifest,
    extract_neu_cls_dataset,
    get_neu_cls_dataset,
    iter_neu_cls_datasets,
    plan_neu_cls_fetch,
)


def _write_fixture_source(source_root: Path) -> None:
    images_root = source_root / "IMAGES"
    images_root.mkdir(parents=True, exist_ok=True)
    for class_code in ("Cr", "RS", "Sc", "In", "Pa", "PS"):
        for index in range(3):
            image = Image.new("L", (64, 64), color=40 + index * 30)
            image.save(images_root / f"{class_code}_{index:03d}.bmp")


def test_neu_cls_metadata_and_aliases() -> None:
    datasets = list(iter_neu_cls_datasets())

    assert datasets == [NEU_CLS_DATASET]
    assert "research-only" in NEU_CLS_LICENCE.lower() or "verify" in NEU_CLS_LICENCE.lower()
    assert get_neu_cls_dataset("neu_cls").name == "shortcut_binary"
    assert get_neu_cls_dataset("all").name == "shortcut_binary"


def test_neu_cls_fetch_plan_requires_manual_archive_when_url_missing(tmp_path: Path) -> None:
    plan = plan_neu_cls_fetch(
        dataset=get_neu_cls_dataset("neu_cls"),
        raw_root=tmp_path,
    )

    assert plan.should_download is False
    assert "archive-url" in plan.reason


def test_neu_cls_prepare_from_manual_source_writes_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "external" / "NEU_CLS"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    _write_fixture_source(source_root)

    extracted_root = extract_neu_cls_dataset(
        raw_root=tmp_path / "data" / "raw",
        interim_root=interim_root,
        source_root=source_root,
    )
    record_count = build_neu_cls_shortcut_manifest(
        extracted_root=extracted_root,
        interim_root=interim_root,
        processed_root=processed_root,
        project_root=tmp_path,
    )

    manifest_path = processed_root / "neu_cls" / "shortcut_binary" / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    assert record_count == len(records)
    assert {record["label"] for record in records} == {"linear_defect", "area_defect"}
    assert {record["variant"] for record in records} >= {
        "correlated",
        "clean",
        "swapped_stamp",
        "no_stamp",
    }
    assert any(record["split"] == "train" for record in records)
    assert any(record["split"] == "test" for record in records)


def test_cli_neu_cls_dry_run_reports_manual_fetch_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "fetch",
            "neu_cls",
            "--category",
            "shortcut_binary",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "manual:" in output
    assert "target dir:" in output
