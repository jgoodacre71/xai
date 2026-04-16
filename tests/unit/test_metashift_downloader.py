from __future__ import annotations

import json
import pickle
from pathlib import Path

from PIL import Image

from xai_demo_suite.cli.data import main
from xai_demo_suite.data.downloaders.metashift import (
    METASHIFT_DATASET,
    METASHIFT_LICENCE,
    build_metashift_manifest,
    get_metashift_dataset,
    iter_metashift_datasets,
    metashift_source_root,
    plan_metashift_fetch,
)


def _write_metashift_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (90, 120, 150)).save(path)


def _write_metashift_fixture(source_root: Path) -> None:
    group_mapping = {
        "cat(indoor)": ["cat_train.jpg", "cat_test.jpg"],
        "dog(outdoor)": ["dog_train.jpg", "dog_test.jpg"],
    }
    (source_root / "train" / "cat").mkdir(parents=True, exist_ok=True)
    (source_root / "train" / "dog").mkdir(parents=True, exist_ok=True)
    (source_root / "val_out_of_domain" / "cat").mkdir(parents=True, exist_ok=True)
    (source_root / "val_out_of_domain" / "dog").mkdir(parents=True, exist_ok=True)
    _write_metashift_image(source_root / "train" / "cat" / "cat_train.jpg")
    _write_metashift_image(source_root / "train" / "dog" / "dog_train.jpg")
    _write_metashift_image(source_root / "val_out_of_domain" / "cat" / "cat_test.jpg")
    _write_metashift_image(source_root / "val_out_of_domain" / "dog" / "dog_test.jpg")
    with (source_root / "imageID_to_group.pkl").open("wb") as output_file:
        pickle.dump(group_mapping, output_file)


def test_metashift_metadata_and_aliases() -> None:
    datasets = list(iter_metashift_datasets())

    assert datasets == [METASHIFT_DATASET]
    assert METASHIFT_LICENCE == "MIT repository plus upstream Visual Genome / GQA terms."
    assert get_metashift_dataset("metashift").name == "subpopulation_shift_cat_dog_indoor_outdoor"


def test_metashift_fetch_plan_is_manual(tmp_path: Path) -> None:
    dataset = get_metashift_dataset("metashift")
    plan = plan_metashift_fetch(dataset=dataset, external_root=tmp_path)

    assert plan.should_download is False
    assert "generation scripts" in plan.reason
    assert plan.target_root == metashift_source_root(tmp_path, dataset)


def test_build_metashift_manifest_from_fixture(tmp_path: Path) -> None:
    dataset = get_metashift_dataset("metashift")
    source_root = metashift_source_root(tmp_path / "external", dataset)
    _write_metashift_fixture(source_root)
    manifest_path = tmp_path / "data" / "processed" / "metashift" / dataset.name / "manifest.jsonl"

    record_count = build_metashift_manifest(
        dataset=dataset,
        source_root=source_root,
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    test_cat = next(
        record
        for record in records
        if record["split"] == "test" and record["label"] == "cat"
    )
    train_dog = next(
        record
        for record in records
        if record["split"] == "train" and record["label"] == "dog"
    )

    assert record_count == 4
    assert test_cat["habitat"] == "indoor"
    assert test_cat["group"] == "cat_indoor"
    assert test_cat["is_aligned"] is True
    assert train_dog["habitat"] == "outdoor"
    assert train_dog["is_aligned"] is True


def test_cli_metashift_dry_run_reports_manual_target(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "fetch",
            "metashift",
            "--category",
            "subpopulation_shift_cat_dog_indoor_outdoor",
            "--external-root",
            str(tmp_path / "external"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "manual:" in output
    assert "target root:" in output
    assert "MetaShift-subpopulation-shift" in output
