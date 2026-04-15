from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.mvtec_loco_ad import (
    MVTEC_LOCO_AD_CATEGORIES,
    MVTEC_LOCO_AD_LICENCE,
    build_mvtec_loco_ad_manifest,
    download_mvtec_loco_ad_category,
    extract_mvtec_loco_ad_category,
    get_mvtec_loco_ad_category,
    iter_mvtec_loco_ad_categories,
    mvtec_loco_ad_archive_path,
    plan_mvtec_loco_ad_fetch,
)


def _write_fixture_archive(archive_path: Path) -> None:
    source_root = archive_path.parent / "fixture"
    category_root = source_root / "juice_bottle"
    (category_root / "train" / "good").mkdir(parents=True)
    (category_root / "validation" / "good").mkdir(parents=True)
    (category_root / "test" / "logical_anomalies").mkdir(parents=True)
    (category_root / "test" / "structural_anomalies").mkdir(parents=True)
    (category_root / "test" / "good").mkdir(parents=True)
    (category_root / "ground_truth" / "logical_anomalies" / "030").mkdir(parents=True)
    (category_root / "ground_truth" / "structural_anomalies").mkdir(parents=True)
    (category_root / "train" / "good" / "000.png").write_bytes(b"train")
    (category_root / "validation" / "good" / "010.png").write_bytes(b"validation")
    (category_root / "test" / "good" / "020.png").write_bytes(b"good")
    (category_root / "test" / "logical_anomalies" / "030.png").write_bytes(b"logical")
    (category_root / "test" / "structural_anomalies" / "040.png").write_bytes(b"structural")
    (category_root / "ground_truth" / "logical_anomalies" / "030" / "000.png").write_bytes(
        b"mask"
    )
    (category_root / "ground_truth" / "structural_anomalies" / "040.png").write_bytes(
        b"mask"
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:xz") as archive:
        archive.add(category_root, arcname="juice_bottle")


def test_mvtec_loco_ad_metadata_covers_official_categories() -> None:
    categories = list(iter_mvtec_loco_ad_categories())

    assert len(categories) == 5
    assert MVTEC_LOCO_AD_LICENCE == "CC BY-NC-SA 4.0"
    assert MVTEC_LOCO_AD_CATEGORIES["juice_bottle"].archive_name == "juice_bottle.tar.xz"
    assert MVTEC_LOCO_AD_CATEGORIES["juice_bottle"].url.endswith("/juice_bottle.tar.xz")


def test_get_loco_category_accepts_display_variants() -> None:
    assert get_mvtec_loco_ad_category("Juice Bottle").name == "juice_bottle"
    assert get_mvtec_loco_ad_category("screw-bag").name == "screw_bag"


def test_loco_fetch_plan_skips_existing_archive(tmp_path: Path) -> None:
    category = get_mvtec_loco_ad_category("juice_bottle")
    archive_path = mvtec_loco_ad_archive_path(category, tmp_path)
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"existing")

    plan = plan_mvtec_loco_ad_fetch(category=category, raw_root=tmp_path)

    assert plan.should_download is False
    assert "already exists" in plan.reason


def test_loco_download_uses_injected_downloader(tmp_path: Path) -> None:
    category = get_mvtec_loco_ad_category("juice_bottle")
    seen: dict[str, Path] = {}

    def fake_downloader(url: str, destination: Path) -> Path:
        seen["url"] = url
        seen["destination"] = destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")
        return destination

    result = download_mvtec_loco_ad_category(
        category=category,
        raw_root=tmp_path,
        downloader=fake_downloader,
    )

    assert result.status == "downloaded"
    assert result.archive_path.exists()
    assert seen["url"] == category.url


def test_loco_extract_and_manifest_from_fixture_archive(tmp_path: Path) -> None:
    category = get_mvtec_loco_ad_category("juice_bottle")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    archive_path = mvtec_loco_ad_archive_path(category, raw_root)
    _write_fixture_archive(archive_path)

    extracted_root = extract_mvtec_loco_ad_category(
        category=category,
        raw_root=raw_root,
        interim_root=interim_root,
    )
    manifest_path = processed_root / "mvtec_loco_ad" / "juice_bottle" / "manifest.jsonl"
    record_count = build_mvtec_loco_ad_manifest(
        category=category,
        category_root=extracted_root / "juice_bottle",
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    logical_record = next(
        record for record in records if record["defect_type"] == "logical_anomalies"
    )
    validation_record = next(record for record in records if record["split"] == "validation")

    assert record_count == 5
    assert logical_record["is_anomalous"] is True
    assert logical_record["mask_path"].endswith("logical_anomalies/030/000.png")
    assert validation_record["is_anomalous"] is False


def test_loco_extract_refuses_existing_interim_without_overwrite(tmp_path: Path) -> None:
    category = get_mvtec_loco_ad_category("juice_bottle")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    archive_path = mvtec_loco_ad_archive_path(category, raw_root)
    _write_fixture_archive(archive_path)

    extract_mvtec_loco_ad_category(category=category, raw_root=raw_root, interim_root=interim_root)

    with pytest.raises(FileExistsError):
        extract_mvtec_loco_ad_category(
            category=category,
            raw_root=raw_root,
            interim_root=interim_root,
        )


def test_loco_cli_dry_run_reports_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "fetch",
            "mvtec_loco_ad",
            "--category",
            "juice_bottle",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "download:" in output
    assert "juice_bottle.tar.xz" in output
