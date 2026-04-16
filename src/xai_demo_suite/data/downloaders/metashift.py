"""MetaShift preparation helpers for the published cat-vs-dog shift split."""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.utils.io import ensure_directory

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

METASHIFT_SOURCE_URL = "https://github.com/Weixin-Liang/MetaShift"
METASHIFT_DOCS_URL = "https://metashift.readthedocs.io/en/latest/sub_pages/applications.html"
METASHIFT_LICENCE = "MIT repository plus upstream Visual Genome / GQA terms."
METASHIFT_USAGE_RESTRICTION = (
    "Treat generated image splits as research-only until upstream Visual Genome / GQA terms "
    "have been checked."
)


@dataclass(frozen=True, slots=True)
class MetaShiftDataset:
    """Metadata for a supported MetaShift split."""

    name: str
    label: str
    expected_root_name: str


@dataclass(frozen=True, slots=True)
class MetaShiftFetchPlan:
    """Resolved fetch guidance for MetaShift."""

    dataset: MetaShiftDataset
    target_root: Path
    should_download: bool
    reason: str


METASHIFT_DATASET = MetaShiftDataset(
    name="subpopulation_shift_cat_dog_indoor_outdoor",
    label="MetaShift Cat vs Dog Indoor/Outdoor",
    expected_root_name="MetaShift-subpopulation-shift",
)


def iter_metashift_datasets() -> Iterable[MetaShiftDataset]:
    """Yield the supported MetaShift dataset entries."""

    yield METASHIFT_DATASET


def get_metashift_dataset(name: str) -> MetaShiftDataset:
    """Return the supported MetaShift dataset metadata."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "metashift": METASHIFT_DATASET,
        "subpopulation_shift_cat_dog_indoor_outdoor": METASHIFT_DATASET,
        "cat_dog_indoor_outdoor": METASHIFT_DATASET,
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        raise ValueError(
            "MetaShift currently supports the published cat-vs-dog indoor/outdoor split only: "
            "subpopulation_shift_cat_dog_indoor_outdoor"
        ) from exc


def metashift_source_root(external_root: Path, dataset: MetaShiftDataset) -> Path:
    """Return the canonical external-data root for a MetaShift split."""

    return external_root / "metashift" / dataset.expected_root_name


def plan_metashift_fetch(
    *,
    dataset: MetaShiftDataset,
    external_root: Path,
) -> MetaShiftFetchPlan:
    """Return manual fetch guidance for MetaShift."""

    target_root = metashift_source_root(external_root, dataset)
    return MetaShiftFetchPlan(
        dataset=dataset,
        target_root=target_root,
        should_download=False,
        reason=(
            "MetaShift requires upstream Visual Genome / GQA assets and generation scripts. "
            "Follow the upstream repository instructions, then place the generated split under "
            f"{target_root} or pass --source-root to prepare."
        ),
    )


def _normalise_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _group_filename_set(values: object) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("MetaShift group mapping values must be sequences of image names.")
    return {Path(str(item)).name for item in values}


def _group_mapping(group_path: Path) -> dict[str, str]:
    with group_path.open("rb") as input_file:
        raw_mapping = pickle.load(input_file)
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"MetaShift group file {group_path} must contain a mapping.")

    filename_to_group: dict[str, str] = {}
    for raw_group_name, raw_values in raw_mapping.items():
        group_name = str(raw_group_name)
        for filename in _group_filename_set(raw_values):
            filename_to_group[filename] = group_name
    return filename_to_group


def _parse_group_name(group_name: str) -> tuple[str, str]:
    normalised = group_name.strip().lower().replace(" ", "")
    if normalised.startswith("cat(") and normalised.endswith(")"):
        return "cat", normalised[4:-1]
    if normalised.startswith("dog(") and normalised.endswith(")"):
        return "dog", normalised[4:-1]
    raise ValueError(f"Unsupported MetaShift group name: {group_name}")


def _canonical_group(label: str, habitat: str) -> str:
    return f"{label}_{habitat}"


def _is_aligned(label: str, habitat: str) -> bool:
    return (label == "cat" and habitat == "indoor") or (
        label == "dog" and habitat == "outdoor"
    )


def _iter_split_images(source_root: Path) -> Iterable[tuple[str, str, Path]]:
    split_aliases = {"train": "train", "val_out_of_domain": "test"}
    for source_split, manifest_split in split_aliases.items():
        split_root = source_root / source_split
        if not split_root.exists():
            continue
        for label_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            label = label_dir.name.lower()
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    yield manifest_split, label, image_path


def build_metashift_manifest(
    *,
    dataset: MetaShiftDataset,
    source_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> int:
    """Write a canonical JSONL manifest for the supported MetaShift split."""

    group_path = source_root / "imageID_to_group.pkl"
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not group_path.exists():
        raise FileNotFoundError(group_path)

    filename_to_group = _group_mapping(group_path)
    records: list[dict[str, object]] = []
    for split_name, label, image_path in _iter_split_images(source_root):
        group_name = filename_to_group.get(image_path.name)
        if group_name is None:
            raise ValueError(
                f"MetaShift image {image_path.name} is missing from imageID_to_group.pkl."
            )
        group_label, habitat = _parse_group_name(group_name)
        if group_label != label:
            raise ValueError(
                f"MetaShift label mismatch for {image_path.name}: folder label {label}, "
                f"group label {group_label}"
            )
        records.append(
            {
                "dataset": "metashift",
                "category": dataset.name,
                "split": split_name,
                "label": label,
                "habitat": habitat,
                "group": _canonical_group(label, habitat),
                "is_aligned": _is_aligned(label, habitat),
                "image_path": _normalise_path(image_path, project_root),
            }
        )

    if not records:
        raise ValueError(f"No MetaShift images discovered under {source_root}.")

    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)
