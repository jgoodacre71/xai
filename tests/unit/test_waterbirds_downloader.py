from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.waterbirds import (
    WATERBIRDS_CATEGORIES,
    WATERBIRDS_LICENCE,
    build_waterbirds_manifest,
    download_waterbirds_category,
    extract_waterbirds_category,
    get_waterbirds_category,
    iter_waterbirds_categories,
    plan_waterbirds_fetch,
    waterbirds_archive_path,
)


def _write_fixture_archive(archive_path: Path) -> None:
    source_root = archive_path.parent / "fixture" / "waterbird_complete95_forest2water2"
    (source_root / "images").mkdir(parents=True)
    (source_root / "images" / "000.jpg").write_bytes(b"train")
    (source_root / "images" / "001.jpg").write_bytes(b"val")
    (source_root / "images" / "002.jpg").write_bytes(b"test")
    (source_root / "metadata.csv").write_text(
        "\n".join(
            [
                "img_filename,y,place,split",
                "images/000.jpg,1,1,0",
                "images/001.jpg,0,0,1",
                "images/002.jpg,1,0,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_root, arcname="waterbird_complete95_forest2water2")


def test_waterbirds_metadata_covers_supported_release() -> None:
    categories = list(iter_waterbirds_categories())

    assert len(categories) == 1
    assert WATERBIRDS_LICENCE == "Verify upstream CUB and Places terms before use."
    assert WATERBIRDS_CATEGORIES["waterbird_complete95_forest2water2"].archive_name.endswith(
        ".tar.gz"
    )


def test_waterbirds_fetch_plan_skips_existing_archive(tmp_path: Path) -> None:
    category = get_waterbirds_category("waterbird_complete95_forest2water2")
    archive_path = waterbirds_archive_path(category, tmp_path)
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"existing")

    plan = plan_waterbirds_fetch(category=category, raw_root=tmp_path)

    assert plan.should_download is False
    assert "already exists" in plan.reason


def test_waterbirds_download_uses_injected_downloader(tmp_path: Path) -> None:
    category = get_waterbirds_category("waterbird_complete95_forest2water2")
    seen: dict[str, Path] = {}

    def fake_downloader(url: str, destination: Path) -> Path:
        seen["url"] = url
        seen["destination"] = destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")
        return destination

    result = download_waterbirds_category(
        category=category,
        raw_root=tmp_path,
        downloader=fake_downloader,
    )

    assert result.status == "downloaded"
    assert result.archive_path.exists()
    assert seen["url"] == category.url


def test_extract_and_manifest_from_fixture_archive(tmp_path: Path) -> None:
    category = get_waterbirds_category("waterbird_complete95_forest2water2")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    archive_path = waterbirds_archive_path(category, raw_root)
    _write_fixture_archive(archive_path)

    extracted_root = extract_waterbirds_category(
        category=category,
        raw_root=raw_root,
        interim_root=interim_root,
    )
    manifest_path = processed_root / "waterbirds" / category.name / "manifest.jsonl"
    record_count = build_waterbirds_manifest(
        category=category,
        category_root=extracted_root / category.name,
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    train_record = next(record for record in records if record["split"] == "train")
    test_record = next(record for record in records if record["split"] == "test")

    assert record_count == 3
    assert train_record["label"] == "waterbird"
    assert train_record["habitat"] == "water"
    assert train_record["is_aligned"] is True
    assert test_record["group"] == "waterbird_on_land"
    assert test_record["is_aligned"] is False


def test_waterbirds_extract_refuses_existing_interim_without_overwrite(tmp_path: Path) -> None:
    category = get_waterbirds_category("waterbird_complete95_forest2water2")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    archive_path = waterbirds_archive_path(category, raw_root)
    _write_fixture_archive(archive_path)

    extract_waterbirds_category(category=category, raw_root=raw_root, interim_root=interim_root)

    with pytest.raises(FileExistsError):
        extract_waterbirds_category(
            category=category,
            raw_root=raw_root,
            interim_root=interim_root,
        )


def test_cli_waterbirds_dry_run_reports_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "fetch",
            "waterbirds",
            "--category",
            "waterbird_complete95_forest2water2",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "download:" in output
    assert "waterbird_complete95_forest2water2.tar.gz" in output
