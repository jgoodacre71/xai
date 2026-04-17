from __future__ import annotations

import json
import zipfile
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


def _write_split_fixture_source(source_root: Path) -> None:
    for split in ("train", "valid"):
        images_root = source_root / split / split / "images"
        images_root.mkdir(parents=True, exist_ok=True)
        for class_name in (
            "crazing",
            "rolled-in_scale",
            "scratches",
            "inclusion",
            "patches",
            "pitted_surface",
        ):
            for index in range(2):
                image = Image.new("RGB", (64, 64), color=(40 + index * 20, 60, 80))
                image.save(images_root / f"{class_name}_{index + 1}.jpg")


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
    assert {record["original_class"] for record in records} == {"Sc", "In"}
    assert {record["variant"] for record in records} >= {
        "correlated",
        "clean",
        "swapped_stamp",
        "no_stamp",
    }
    assert any(record["split"] == "train" for record in records)
    assert any(record["split"] == "test" for record in records)


def test_neu_cls_extracts_zip_archive_without_zip_suffix(tmp_path: Path) -> None:
    source_root = tmp_path / "external" / "NEU_CLS"
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    _write_fixture_source(source_root)

    archive_path = raw_root / "neu_cls" / "archives" / "54094775"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for file_path in sorted(source_root.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(source_root))

    extracted_root = extract_neu_cls_dataset(
        raw_root=raw_root,
        interim_root=interim_root,
    )

    assert (extracted_root / "IMAGES" / "Cr_000.bmp").exists()


def test_neu_cls_prepare_accepts_split_layout_archive(tmp_path: Path) -> None:
    source_root = tmp_path / "external" / "neu_split"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    _write_split_fixture_source(source_root)

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
    assert any(record["split"] == "train" for record in records)
    assert any(record["split"] == "test" for record in records)
    assert {record["label"] for record in records} == {"linear_defect", "area_defect"}
    assert {record["original_class"] for record in records} == {"Sc", "In"}


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
