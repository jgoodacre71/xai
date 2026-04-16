from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.mvtec_ad_2 import (
    MVTEC_AD_2_DATASET,
    MVTEC_AD_2_LICENCE,
    build_mvtec_ad_2_manifests,
    download_mvtec_ad_2_dataset,
    extract_mvtec_ad_2_dataset,
    get_mvtec_ad_2_dataset,
    iter_mvtec_ad_2_datasets,
    mvtec_ad_2_archive_dir,
    plan_mvtec_ad_2_fetch,
)


def _write_fixture_archive(archive_path: Path) -> None:
    source_root = archive_path.parent / "fixture" / "mvtec_ad_2"
    for scenario in ("cable_gland", "sheet_metal"):
        (source_root / scenario / "train" / "good").mkdir(parents=True, exist_ok=True)
        (source_root / scenario / "validation" / "good").mkdir(parents=True, exist_ok=True)
        (source_root / scenario / "test_public" / "good").mkdir(parents=True, exist_ok=True)
        (source_root / scenario / "test_public" / "crack").mkdir(parents=True, exist_ok=True)
        (source_root / scenario / "test_private" / "good").mkdir(parents=True, exist_ok=True)
        (source_root / scenario / "test_private" / "crack").mkdir(parents=True, exist_ok=True)
        (source_root / scenario / "ground_truth_public" / "crack").mkdir(
            parents=True,
            exist_ok=True,
        )
        (source_root / scenario / "train" / "good" / "000.png").write_bytes(b"train")
        (source_root / scenario / "validation" / "good" / "001.png").write_bytes(b"val")
        (source_root / scenario / "test_public" / "good" / "002.png").write_bytes(b"good")
        (source_root / scenario / "test_public" / "crack" / "003.png").write_bytes(b"public")
        (source_root / scenario / "test_private" / "good" / "004.png").write_bytes(b"good")
        (source_root / scenario / "test_private" / "crack" / "005.png").write_bytes(
            b"private"
        )
        (source_root / scenario / "ground_truth_public" / "crack" / "003_mask.png").write_bytes(
            b"mask"
        )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:xz") as archive:
        archive.add(source_root, arcname="mvtec_ad_2")


def test_mvtec_ad_2_metadata_and_aliases() -> None:
    datasets = list(iter_mvtec_ad_2_datasets())

    assert datasets == [MVTEC_AD_2_DATASET]
    assert MVTEC_AD_2_LICENCE == "CC BY-NC-SA 4.0"
    assert get_mvtec_ad_2_dataset("mvtec_ad_2").name == "all"
    assert get_mvtec_ad_2_dataset("all").scenario_count == 8


def test_mvtec_ad_2_fetch_plan_requires_manual_url_when_archive_missing(tmp_path: Path) -> None:
    plan = plan_mvtec_ad_2_fetch(
        dataset=get_mvtec_ad_2_dataset("all"),
        raw_root=tmp_path,
    )

    assert plan.should_download is False
    assert "page-gated" in plan.reason


def test_mvtec_ad_2_download_uses_injected_downloader(tmp_path: Path) -> None:
    dataset = get_mvtec_ad_2_dataset("all")
    seen: dict[str, Path] = {}
    archive_url = "https://example.com/mvtec_ad_2_fixture.tar.xz"

    def fake_downloader(url: str, destination: Path) -> Path:
        seen["url"] = url
        seen["destination"] = destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")
        return destination

    result = download_mvtec_ad_2_dataset(
        dataset=dataset,
        raw_root=tmp_path,
        archive_url=archive_url,
        downloader=fake_downloader,
    )

    assert result.status == "downloaded"
    assert seen["url"] == archive_url
    assert result.archive_path == mvtec_ad_2_archive_dir(tmp_path) / "mvtec_ad_2_fixture.tar.xz"


def test_extract_and_build_manifests_from_fixture_archive(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    archive_path = mvtec_ad_2_archive_dir(raw_root) / "mvtec_ad_2_fixture.tar.xz"
    _write_fixture_archive(archive_path)

    extracted_root = extract_mvtec_ad_2_dataset(
        raw_root=raw_root,
        interim_root=interim_root,
    )
    scenario_counts = build_mvtec_ad_2_manifests(
        extracted_root=extracted_root,
        processed_root=processed_root,
        project_root=tmp_path,
    )

    assert scenario_counts == {"cable_gland": 6, "sheet_metal": 6}
    manifest_path = processed_root / "mvtec_ad_2" / "cable_gland" / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    public_anomaly = next(
        record
        for record in records
        if record["split"] == "test_public" and record["defect_type"] == "crack"
    )
    private_anomaly = next(
        record
        for record in records
        if record["split"] == "test_private" and record["defect_type"] == "crack"
    )

    assert public_anomaly["mask_path"].endswith("003_mask.png")
    assert private_anomaly["mask_path"] is None


def test_mvtec_ad_2_extract_refuses_existing_interim_without_overwrite(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    archive_path = mvtec_ad_2_archive_dir(raw_root) / "mvtec_ad_2_fixture.tar.xz"
    _write_fixture_archive(archive_path)

    extract_mvtec_ad_2_dataset(raw_root=raw_root, interim_root=interim_root)

    with pytest.raises(FileExistsError):
        extract_mvtec_ad_2_dataset(raw_root=raw_root, interim_root=interim_root)


def test_cli_mvtec_ad_2_dry_run_reports_manual_fetch_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "fetch",
            "mvtec_ad_2",
            "--category",
            "all",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "manual:" in output
    assert "target dir:" in output
