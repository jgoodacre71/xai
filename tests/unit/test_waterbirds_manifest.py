from __future__ import annotations

import json
from pathlib import Path

from xai_demo_suite.data.waterbirds_manifest import (
    filter_waterbirds_records,
    load_waterbirds_manifest,
)


def test_load_waterbirds_manifest_and_filter(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data" / "processed" / "waterbirds" / "demo" / "manifest.jsonl"
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"placeholder")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": "waterbirds",
            "category": "demo",
            "split": "train",
            "label": "waterbird",
            "habitat": "water",
            "group": "waterbird_on_water",
            "is_aligned": True,
            "image_path": image_path.as_posix(),
        },
        {
            "dataset": "waterbirds",
            "category": "demo",
            "split": "test",
            "label": "landbird",
            "habitat": "water",
            "group": "landbird_on_water",
            "is_aligned": False,
            "image_path": image_path.as_posix(),
        },
    ]
    with manifest_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")

    records = load_waterbirds_manifest(manifest_path)

    assert len(records) == 2
    assert records[0].sample_id == "waterbirds/demo/train/sample"
    filtered = filter_waterbirds_records(records, split="test", aligned=False)
    assert len(filtered) == 1
    assert filtered[0].group == "landbird_on_water"
