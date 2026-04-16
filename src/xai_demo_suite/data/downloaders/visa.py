"""VisA dataset download and preparation helpers."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from xai_demo_suite.utils.io import ensure_directory

VisADownloader = Callable[[str, Path], Path]

VISA_SOURCE_URL = "https://github.com/amazon-science/spot-diff"
VISA_DATASET_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
VISA_LICENCE = "CC BY 4.0"
VISA_LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
VISA_ARCHIVE_NAME = "VisA_20220922.tar"
VISA_SPLIT_CSV_NAME = "1cls.csv"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class VisADataset:
    """Metadata for the VisA dataset release."""

    name: str
    label: str
    subset_count: int
    archive_name: str
    url: str
    split_csv_url: str


@dataclass(frozen=True, slots=True)
class VisAFetchPlan:
    """A resolved download decision for the VisA archive."""

    dataset: VisADataset
    archive_path: Path
    url: str
    should_download: bool
    reason: str


@dataclass(frozen=True, slots=True)
class VisADownloadResult:
    """Result of a raw VisA archive fetch operation."""

    dataset: VisADataset
    archive_path: Path
    status: str


VISA_DATASET = VisADataset(
    name="all",
    label="VisA",
    subset_count=12,
    archive_name=VISA_ARCHIVE_NAME,
    url=VISA_DATASET_URL,
    split_csv_url=VISA_SPLIT_CSV_URL,
)


def iter_visa_datasets() -> Iterable[VisADataset]:
    """Yield the supported VisA dataset entries."""

    yield VISA_DATASET


def get_visa_dataset(name: str) -> VisADataset:
    """Return the supported VisA dataset metadata."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalised not in {"all", "visa"}:
        raise ValueError("VisA currently supports the full dataset bundle only: all")
    return VISA_DATASET


def visa_archive_path(dataset: VisADataset, raw_root: Path) -> Path:
    """Return the canonical raw VisA archive path."""

    return raw_root / "visa" / "archives" / dataset.archive_name


def visa_split_csv_path(raw_root: Path) -> Path:
    """Return the canonical local path for the 1-class split CSV."""

    return raw_root / "visa" / "splits" / VISA_SPLIT_CSV_NAME


def plan_visa_fetch(
    *,
    dataset: VisADataset,
    raw_root: Path,
    overwrite: bool = False,
) -> VisAFetchPlan:
    """Resolve whether the VisA archive should be downloaded."""

    archive_path = visa_archive_path(dataset, raw_root)
    if archive_path.exists() and not overwrite:
        return VisAFetchPlan(
            dataset=dataset,
            archive_path=archive_path,
            url=dataset.url,
            should_download=False,
            reason="archive already exists; use --overwrite to replace it",
        )
    return VisAFetchPlan(
        dataset=dataset,
        archive_path=archive_path,
        url=dataset.url,
        should_download=True,
        reason="archive missing" if not archive_path.exists() else "overwrite requested",
    )


def stream_download(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` via a temporary partial file."""

    ensure_directory(destination.parent)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "xai-demo-suite/0.1"})
    with urlopen(request, timeout=60) as response, partial_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    os.replace(partial_path, destination)
    return destination


def download_visa_dataset(
    *,
    dataset: VisADataset,
    raw_root: Path,
    overwrite: bool = False,
    downloader: VisADownloader = stream_download,
) -> VisADownloadResult:
    """Fetch the VisA archive into ``data/raw``."""

    plan = plan_visa_fetch(dataset=dataset, raw_root=raw_root, overwrite=overwrite)
    if not plan.should_download:
        return VisADownloadResult(
            dataset=dataset,
            archive_path=plan.archive_path,
            status="exists",
        )
    downloader(plan.url, plan.archive_path)
    return VisADownloadResult(
        dataset=dataset,
        archive_path=plan.archive_path,
        status="downloaded",
    )


def download_visa_split_csv(
    *,
    raw_root: Path,
    overwrite: bool = False,
    downloader: VisADownloader = stream_download,
) -> Path:
    """Fetch the upstream 1-class split CSV used for VisA preparation."""

    destination = visa_split_csv_path(raw_root)
    if destination.exists() and not overwrite:
        return destination
    return downloader(VISA_SPLIT_CSV_URL, destination)


def _safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    destination_root = destination.resolve()
    safe_members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target_path = (destination / member.name).resolve()
        if destination_root != target_path and destination_root not in target_path.parents:
            raise ValueError(f"Archive member escapes extraction directory: {member.name}")
        safe_members.append(member)
    return safe_members


def extract_visa_dataset(
    *,
    dataset: VisADataset,
    raw_root: Path,
    interim_root: Path,
    overwrite: bool = False,
) -> Path:
    """Extract the raw VisA archive into ``data/interim``."""

    archive_path = visa_archive_path(dataset, raw_root)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing raw archive {archive_path}. Run fetch before prepare.")

    extract_root = interim_root / "visa" / "raw"
    if extract_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{extract_root} already exists; use --overwrite to replace extracted data."
            )
        shutil.rmtree(extract_root)

    ensure_directory(extract_root)
    with tarfile.open(archive_path, mode="r:*") as archive:
        archive.extractall(
            path=extract_root,
            members=_safe_members(archive, extract_root),
            filter="data",
        )
    return extract_root


def _normalise_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _binarise_mask(source_path: Path, destination: Path) -> Path:
    ensure_directory(destination.parent)
    with Image.open(source_path) as mask_image:
        values = np.asarray(mask_image)
        values = np.where(values != 0, 255, 0).astype(np.uint8)
        Image.fromarray(values).save(destination)
    return destination


def _resolve_visa_root(extracted_root: Path) -> Path:
    visa_root = extracted_root / "VisA"
    if visa_root.exists():
        return visa_root
    child_dirs = [path for path in extracted_root.iterdir() if path.is_dir()]
    if len(child_dirs) == 1:
        return child_dirs[0]
    return extracted_root


def prepare_visa_one_class_layout(
    *,
    extracted_root: Path,
    interim_root: Path,
    split_csv_path: Path,
    overwrite: bool = False,
) -> Path:
    """Reorganise raw VisA data into a MVTec-like 1-class layout."""

    if not split_csv_path.exists():
        raise FileNotFoundError(f"Missing VisA split CSV {split_csv_path}.")

    visa_root = _resolve_visa_root(extracted_root)
    prepared_root = interim_root / "visa" / "1cls"
    if prepared_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{prepared_root} already exists; use --overwrite to replace prepared data."
            )
        shutil.rmtree(prepared_root)

    ensure_directory(prepared_root)
    with split_csv_path.open("r", encoding="utf-8") as split_file:
        reader = csv.DictReader(split_file)
        for row in reader:
            object_name = str(row["object"])
            split_name = str(row["set"])
            label_name = "good" if str(row["label"]) == "normal" else "bad"
            image_path = visa_root / str(row["image_path"])
            mask_path = visa_root / str(row["mask_path"])
            image_name = Path(str(row["image_path"])).name
            mask_name = Path(str(row["mask_path"])).name

            destination_image = prepared_root / object_name / split_name / label_name / image_name
            ensure_directory(destination_image.parent)
            shutil.copyfile(image_path, destination_image)

            if split_name == "test" and label_name == "bad":
                destination_mask = (
                    prepared_root / object_name / "ground_truth" / label_name / mask_name
                )
                _binarise_mask(mask_path, destination_mask)
    return prepared_root


def _iter_prepared_images(category_root: Path) -> Iterable[tuple[str, str, Path]]:
    for split_name in ("train", "test"):
        split_root = category_root / split_name
        if not split_root.exists():
            continue
        for defect_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            for image_path in sorted(defect_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    yield split_name, defect_type, image_path


def _resolve_mask_path(category_root: Path, defect_type: str, image_path: Path) -> Path | None:
    ground_truth_root = category_root / "ground_truth" / defect_type
    if not ground_truth_root.exists():
        return None

    exact_path = ground_truth_root / image_path.name
    if exact_path.exists():
        return exact_path

    for candidate in sorted(ground_truth_root.iterdir()):
        if candidate.is_file() and candidate.stem == image_path.stem:
            return candidate
    return None


def build_visa_manifest(
    *,
    category_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> int:
    """Write a JSONL manifest for one prepared VisA category."""

    ensure_directory(manifest_path.parent)
    records: list[dict[str, object]] = []
    for split_name, defect_type, image_path in _iter_prepared_images(category_root):
        is_anomalous = defect_type != "good"
        mask_path = None
        if split_name == "test" and is_anomalous:
            mask_path = _resolve_mask_path(category_root, defect_type, image_path)
        records.append(
            {
                "dataset": "visa",
                "category": category_root.name,
                "split": split_name,
                "defect_type": defect_type,
                "is_anomalous": is_anomalous,
                "image_path": _normalise_path(image_path, project_root),
                "mask_path": _normalise_path(mask_path, project_root) if mask_path else None,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)


def build_visa_manifests(
    *,
    prepared_root: Path,
    processed_root: Path,
    project_root: Path,
) -> dict[str, int]:
    """Write one manifest per prepared VisA category and return record counts."""

    category_counts: dict[str, int] = {}
    for category_root in sorted(path for path in prepared_root.iterdir() if path.is_dir()):
        manifest_path = processed_root / "visa" / category_root.name / "manifest.jsonl"
        category_counts[category_root.name] = build_visa_manifest(
            category_root=category_root,
            manifest_path=manifest_path,
            project_root=project_root,
        )
    if not category_counts:
        raise ValueError(f"No prepared VisA categories discovered under {prepared_root}.")
    return category_counts
