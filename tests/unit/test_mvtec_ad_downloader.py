from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.mvtec_ad import (
    MVTEC_AD_CATEGORIES,
    MVTEC_AD_LICENCE,
    build_mvtec_ad_manifest,
    download_mvtec_ad_category,
    extract_mvtec_ad_category,
    get_mvtec_ad_category,
    iter_mvtec_ad_categories,
    mvtec_ad_archive_path,
    plan_mvtec_ad_fetch,
)


def _write_fixture_archive(archive_path: Path) -> None:
    source_root = archive_path.parent / "fixture"
    (source_root / "bottle" / "train" / "good").mkdir(parents=True)
    (source_root / "bottle" / "test" / "crack").mkdir(parents=True)
    (source_root / "bottle" / "test" / "good").mkdir(parents=True)
    (source_root / "bottle" / "ground_truth" / "crack").mkdir(parents=True)
    (source_root / "bottle" / "train" / "good" / "000.png").write_bytes(b"train")
    (source_root / "bottle" / "test" / "crack" / "001.png").write_bytes(b"test")
    (source_root / "bottle" / "test" / "good" / "002.png").write_bytes(b"good")
    (source_root / "bottle" / "ground_truth" / "crack" / "001_mask.png").write_bytes(
        b"mask"
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:xz") as archive:
        archive.add(source_root / "bottle", arcname="bottle")


def test_mvtec_ad_metadata_covers_official_categories() -> None:
    categories = list(iter_mvtec_ad_categories())

    assert len(categories) == 15
    assert MVTEC_AD_LICENCE == "CC BY-NC-SA 4.0"
    assert MVTEC_AD_CATEGORIES["bottle"].archive_name == "bottle.tar.xz"
    assert MVTEC_AD_CATEGORIES["bottle"].url.endswith("/bottle.tar.xz")


def test_get_category_accepts_display_variants() -> None:
    assert get_mvtec_ad_category("Metal Nut").name == "metal_nut"
    assert get_mvtec_ad_category("metal-nut").name == "metal_nut"


def test_fetch_plan_skips_existing_archive(tmp_path: Path) -> None:
    category = get_mvtec_ad_category("bottle")
    archive_path = mvtec_ad_archive_path(category, tmp_path)
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"existing")

    plan = plan_mvtec_ad_fetch(category=category, raw_root=tmp_path)

    assert plan.should_download is False
    assert "already exists" in plan.reason


def test_download_uses_injected_downloader(tmp_path: Path) -> None:
    category = get_mvtec_ad_category("bottle")
    seen: dict[str, Path] = {}

    def fake_downloader(url: str, destination: Path) -> Path:
        seen["url"] = url
        seen["destination"] = destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")
        return destination

    result = download_mvtec_ad_category(
        category=category,
        raw_root=tmp_path,
        downloader=fake_downloader,
    )

    assert result.status == "downloaded"
    assert result.archive_path.exists()
    assert seen["url"] == category.url


def test_extract_and_manifest_from_fixture_archive(tmp_path: Path) -> None:
    category = get_mvtec_ad_category("bottle")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    archive_path = mvtec_ad_archive_path(category, raw_root)
    _write_fixture_archive(archive_path)

    extracted_root = extract_mvtec_ad_category(
        category=category,
        raw_root=raw_root,
        interim_root=interim_root,
    )
    manifest_path = processed_root / "mvtec_ad" / "bottle" / "manifest.jsonl"
    record_count = build_mvtec_ad_manifest(
        category=category,
        category_root=extracted_root / "bottle",
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    anomaly_record = next(record for record in records if record["defect_type"] == "crack")

    assert record_count == 3
    assert anomaly_record["is_anomalous"] is True
    assert anomaly_record["mask_path"].endswith("001_mask.png")


def test_extract_refuses_existing_interim_without_overwrite(tmp_path: Path) -> None:
    category = get_mvtec_ad_category("bottle")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    archive_path = mvtec_ad_archive_path(category, raw_root)
    _write_fixture_archive(archive_path)

    extract_mvtec_ad_category(category=category, raw_root=raw_root, interim_root=interim_root)

    with pytest.raises(FileExistsError):
        extract_mvtec_ad_category(
            category=category,
            raw_root=raw_root,
            interim_root=interim_root,
        )


def test_cli_dry_run_reports_target(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["fetch", "mvtec_ad", "--category", "bottle", "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "download:" in output
    assert "bottle.tar.xz" in output
