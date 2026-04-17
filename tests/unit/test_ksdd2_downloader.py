from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.ksdd2 import (
    KSDD2_DATASET,
    KSDD2_LICENCE,
    build_ksdd2_shortcut_manifest,
    extract_ksdd2_dataset,
    get_ksdd2_dataset,
    iter_ksdd2_datasets,
    plan_ksdd2_fetch,
)


def _write_fixture_source(source_root: Path) -> None:
    for split in ("train", "test"):
        for label in ("good", "defect"):
            image_root = source_root / split / label
            image_root.mkdir(parents=True, exist_ok=True)
            for index in range(3):
                colour = (60 + index * 10, 90, 110) if label == "good" else (130, 70, 60)
                Image.new("RGB", (96, 48), colour).save(image_root / f"{label}_{index:03d}.png")


def test_ksdd2_metadata_and_aliases() -> None:
    datasets = list(iter_ksdd2_datasets())

    assert datasets == [KSDD2_DATASET]
    assert "noncommercial" in KSDD2_LICENCE.lower()
    assert get_ksdd2_dataset("ksdd2").name == "shortcut_binary"
    assert get_ksdd2_dataset("all").name == "shortcut_binary"


def test_ksdd2_fetch_plan_requires_manual_archive_when_url_missing(tmp_path: Path) -> None:
    plan = plan_ksdd2_fetch(
        dataset=get_ksdd2_dataset("ksdd2"),
        raw_root=tmp_path,
    )

    assert plan.should_download is False
    assert "archive-url" in plan.reason


def test_ksdd2_prepare_from_manual_source_writes_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "external" / "KolektorSDD2"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    _write_fixture_source(source_root)

    extracted_root = extract_ksdd2_dataset(
        raw_root=tmp_path / "data" / "raw",
        interim_root=interim_root,
        source_root=source_root,
    )
    record_count = build_ksdd2_shortcut_manifest(
        extracted_root=extracted_root,
        interim_root=interim_root,
        processed_root=processed_root,
        project_root=tmp_path,
    )

    manifest_path = processed_root / "ksdd2" / "shortcut_binary" / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    assert record_count == len(records)
    assert {record["label"] for record in records} == {"nominal_surface", "defective_surface"}
    assert {record["variant"] for record in records} >= {
        "correlated",
        "clean",
        "swapped_stamp",
        "no_stamp",
    }
    assert any(record["split"] == "train" for record in records)
    assert any(record["split"] == "test" for record in records)


def test_ksdd2_extracts_zip_archive_without_zip_suffix(tmp_path: Path) -> None:
    source_root = tmp_path / "external" / "KolektorSDD2"
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    _write_fixture_source(source_root)

    archive_path = raw_root / "ksdd2" / "archives" / "ksdd2_bundle"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for file_path in sorted(source_root.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(source_root))

    extracted_root = extract_ksdd2_dataset(
        raw_root=raw_root,
        interim_root=interim_root,
    )

    assert (extracted_root / "train" / "good" / "good_000.png").exists()


def test_cli_ksdd2_dry_run_reports_manual_fetch_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "fetch",
            "ksdd2",
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
