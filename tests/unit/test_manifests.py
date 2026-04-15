from __future__ import annotations

import json
from pathlib import Path

from xai_demo_suite.data.manifests import filter_manifest_records, load_image_manifest


def test_load_image_manifest_resolves_repo_relative_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data" / "processed" / "mvtec_ad" / "bottle" / "manifest.jsonl"
    image_path = (
        tmp_path
        / "data"
        / "interim"
        / "mvtec_ad"
        / "bottle"
        / "train"
        / "good"
        / "000.png"
    )
    manifest_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "mvtec_ad",
                "category": "bottle",
                "split": "train",
                "defect_type": "good",
                "is_anomalous": False,
                "image_path": "data/interim/mvtec_ad/bottle/train/good/000.png",
                "mask_path": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_image_manifest(manifest_path)

    assert len(records) == 1
    assert records[0].image_path == image_path
    assert records[0].sample_id == "mvtec_ad/bottle/train/good/000"


def test_filter_manifest_records(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data" / "processed" / "mvtec_ad" / "bottle" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    rows = [
        {
            "dataset": "mvtec_ad",
            "category": "bottle",
            "split": "train",
            "defect_type": "good",
            "is_anomalous": False,
            "image_path": "data/interim/mvtec_ad/bottle/train/good/000.png",
            "mask_path": None,
        },
        {
            "dataset": "mvtec_ad",
            "category": "bottle",
            "split": "test",
            "defect_type": "broken",
            "is_anomalous": True,
            "image_path": "data/interim/mvtec_ad/bottle/test/broken/001.png",
            "mask_path": "data/interim/mvtec_ad/bottle/ground_truth/broken/001_mask.png",
        },
    ]
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    records = load_image_manifest(manifest_path)

    assert len(filter_manifest_records(records, split="train", defect_type="good")) == 1
    assert len(filter_manifest_records(records, is_anomalous=True)) == 1
