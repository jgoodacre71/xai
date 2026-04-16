from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.visa import (
    VISA_DATASET,
    VISA_LICENCE,
    build_visa_manifests,
    download_visa_dataset,
    extract_visa_dataset,
    get_visa_dataset,
    iter_visa_datasets,
    plan_visa_fetch,
    prepare_visa_one_class_layout,
    visa_archive_path,
    visa_split_csv_path,
)


def _write_fixture_archive(archive_path: Path) -> None:
    source_root = archive_path.parent / "fixture" / "VisA"
    category_root = source_root / "candle"
    (category_root / "Data" / "Images").mkdir(parents=True, exist_ok=True)
    (category_root / "Masks").mkdir(parents=True, exist_ok=True)
    (category_root / "Data" / "Images" / "normal_000.JPG").write_bytes(b"train")
    (category_root / "Data" / "Images" / "normal_001.JPG").write_bytes(b"test-good")
    (category_root / "Data" / "Images" / "anomaly_000.JPG").write_bytes(b"test-bad")
    Image.new("L", (4, 4), color=255).save(category_root / "Masks" / "anomaly_000.png")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w") as archive:
        archive.add(source_root, arcname="VisA")


def _write_split_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "object,set,label,image_path,mask_path",
                "candle,train,normal,candle/Data/Images/normal_000.JPG,",
                "candle,test,normal,candle/Data/Images/normal_001.JPG,",
                "candle,test,anomaly,candle/Data/Images/anomaly_000.JPG,candle/Masks/anomaly_000.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_visa_metadata_and_aliases() -> None:
    datasets = list(iter_visa_datasets())

    assert datasets == [VISA_DATASET]
    assert VISA_LICENCE == "CC BY 4.0"
    assert get_visa_dataset("visa").name == "all"
    assert get_visa_dataset("all").subset_count == 12


def test_visa_fetch_plan_skips_existing_archive(tmp_path: Path) -> None:
    dataset = get_visa_dataset("all")
    archive_path = visa_archive_path(dataset, tmp_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"existing")

    plan = plan_visa_fetch(dataset=dataset, raw_root=tmp_path)

    assert plan.should_download is False
    assert "already exists" in plan.reason


def test_visa_download_uses_injected_downloader(tmp_path: Path) -> None:
    dataset = get_visa_dataset("all")
    seen: dict[str, Path] = {}

    def fake_downloader(url: str, destination: Path) -> Path:
        seen["url"] = url
        seen["destination"] = destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")
        return destination

    result = download_visa_dataset(
        dataset=dataset,
        raw_root=tmp_path,
        downloader=fake_downloader,
    )

    assert result.status == "downloaded"
    assert result.archive_path.exists()
    assert seen["url"] == dataset.url


def test_extract_prepare_and_manifest_from_fixture_archive(tmp_path: Path) -> None:
    dataset = get_visa_dataset("all")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    processed_root = tmp_path / "data" / "processed"
    archive_path = visa_archive_path(dataset, raw_root)
    split_csv_file = visa_split_csv_path(raw_root)
    _write_fixture_archive(archive_path)
    _write_split_csv(split_csv_file)

    extracted_root = extract_visa_dataset(
        dataset=dataset,
        raw_root=raw_root,
        interim_root=interim_root,
    )
    prepared_root = prepare_visa_one_class_layout(
        extracted_root=extracted_root,
        interim_root=interim_root,
        split_csv_path=split_csv_file,
    )
    category_counts = build_visa_manifests(
        prepared_root=prepared_root,
        processed_root=processed_root,
        project_root=tmp_path,
    )

    assert category_counts == {"candle": 3}
    manifest_path = processed_root / "visa" / "candle" / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    anomaly_record = next(record for record in records if record["defect_type"] == "bad")
    good_record = next(
        record
        for record in records
        if record["split"] == "test" and record["defect_type"] == "good"
    )
    mask_path = tmp_path / anomaly_record["mask_path"]

    assert anomaly_record["is_anomalous"] is True
    assert good_record["is_anomalous"] is False
    assert mask_path.name == "anomaly_000.png"
    with Image.open(mask_path) as mask_image:
        assert set(np.asarray(mask_image).ravel()) == {255}


def test_visa_extract_refuses_existing_interim_without_overwrite(tmp_path: Path) -> None:
    dataset = get_visa_dataset("all")
    raw_root = tmp_path / "data" / "raw"
    interim_root = tmp_path / "data" / "interim"
    archive_path = visa_archive_path(dataset, raw_root)
    _write_fixture_archive(archive_path)

    extract_visa_dataset(dataset=dataset, raw_root=raw_root, interim_root=interim_root)

    with pytest.raises(FileExistsError):
        extract_visa_dataset(dataset=dataset, raw_root=raw_root, interim_root=interim_root)


def test_cli_visa_dry_run_reports_archive_and_split_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "fetch",
            "visa",
            "--category",
            "all",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "download:" in output
    assert "VisA_20220922.tar" in output
    assert "split csv:" in output
