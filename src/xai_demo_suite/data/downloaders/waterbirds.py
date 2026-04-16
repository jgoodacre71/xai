"""Waterbirds download and preparation helpers.

The Stanford group DRO repository distributes a tarball of the derived
Waterbirds dataset. This adapter keeps the same raw/interim/processed policy as
the MVTec helpers and writes a canonical JSONL manifest for later real Demo 01
work.
"""

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

from xai_demo_suite.utils.io import ensure_directory

WaterbirdsDownloader = Callable[[str, Path], Path]

WATERBIRDS_SOURCE_URL = "https://github.com/kohpangwei/group_DRO"
WATERBIRDS_DOWNLOADS_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
WATERBIRDS_LICENCE = "Verify upstream CUB and Places terms before use."
WATERBIRDS_USAGE_RESTRICTION = "Treat as research-only unless upstream terms clearly allow more."


@dataclass(frozen=True, slots=True)
class WaterbirdsCategory:
    """Download metadata for the standard Waterbirds release."""

    name: str
    label: str
    archive_name: str
    url: str


@dataclass(frozen=True, slots=True)
class WaterbirdsFetchPlan:
    """Resolved download decision for the Waterbirds archive."""

    category: WaterbirdsCategory
    archive_path: Path
    url: str
    should_download: bool
    reason: str


@dataclass(frozen=True, slots=True)
class WaterbirdsDownloadResult:
    """Result of a Waterbirds archive fetch."""

    category: WaterbirdsCategory
    archive_path: Path
    status: str


WATERBIRDS_CATEGORIES: dict[str, WaterbirdsCategory] = {
    "waterbird_complete95_forest2water2": WaterbirdsCategory(
        name="waterbird_complete95_forest2water2",
        label="Waterbirds Complete 95",
        archive_name="waterbird_complete95_forest2water2.tar.gz",
        url=WATERBIRDS_DOWNLOADS_URL,
    )
}


def iter_waterbirds_categories() -> Iterable[WaterbirdsCategory]:
    """Yield supported Waterbirds releases in a stable order."""

    for name in sorted(WATERBIRDS_CATEGORIES):
        yield WATERBIRDS_CATEGORIES[name]


def get_waterbirds_category(name: str) -> WaterbirdsCategory:
    """Return a Waterbirds release by canonical or display-like name."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return WATERBIRDS_CATEGORIES[normalised]
    except KeyError as exc:
        valid = ", ".join(sorted(WATERBIRDS_CATEGORIES))
        raise ValueError(
            f"Unknown Waterbirds category '{name}'. Valid categories: {valid}"
        ) from exc


def waterbirds_archive_path(category: WaterbirdsCategory, raw_root: Path) -> Path:
    """Return the raw archive location for a Waterbirds release."""

    return raw_root / "waterbirds" / "archives" / category.archive_name


def plan_waterbirds_fetch(
    category: WaterbirdsCategory,
    raw_root: Path,
    overwrite: bool = False,
) -> WaterbirdsFetchPlan:
    """Resolve whether the Waterbirds archive should be downloaded."""

    archive_path = waterbirds_archive_path(category=category, raw_root=raw_root)
    if archive_path.exists() and not overwrite:
        return WaterbirdsFetchPlan(
            category=category,
            archive_path=archive_path,
            url=category.url,
            should_download=False,
            reason="archive already exists; use --overwrite to replace it",
        )
    return WaterbirdsFetchPlan(
        category=category,
        archive_path=archive_path,
        url=category.url,
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


def download_waterbirds_category(
    category: WaterbirdsCategory,
    raw_root: Path,
    overwrite: bool = False,
    downloader: WaterbirdsDownloader = stream_download,
) -> WaterbirdsDownloadResult:
    """Fetch the Waterbirds archive into ``data/raw`` without implicit overwrites."""

    plan = plan_waterbirds_fetch(category=category, raw_root=raw_root, overwrite=overwrite)
    if not plan.should_download:
        return WaterbirdsDownloadResult(
            category=category,
            archive_path=plan.archive_path,
            status="exists",
        )

    if plan.archive_path.exists() and not overwrite:
        raise FileExistsError(plan.archive_path)
    downloader(plan.url, plan.archive_path)
    return WaterbirdsDownloadResult(
        category=category,
        archive_path=plan.archive_path,
        status="downloaded",
    )


def _safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    destination_root = destination.resolve()
    safe_members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target_path = (destination / member.name).resolve()
        if destination_root != target_path and destination_root not in target_path.parents:
            raise ValueError(f"Archive member escapes extraction directory: {member.name}")
        safe_members.append(member)
    return safe_members


def extract_waterbirds_category(
    category: WaterbirdsCategory,
    raw_root: Path,
    interim_root: Path,
    overwrite: bool = False,
) -> Path:
    """Extract a raw Waterbirds archive into ``data/interim`` and return its root."""

    archive_path = waterbirds_archive_path(category=category, raw_root=raw_root)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing raw archive {archive_path}. Run fetch before prepare.")

    extract_root = interim_root / "waterbirds"
    category_root = extract_root / category.name
    if category_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{category_root} already exists; use --overwrite to replace extracted data."
            )
        shutil.rmtree(category_root)

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


def _label_name(value: str) -> str:
    if value in {"1", "waterbird"}:
        return "waterbird"
    if value in {"0", "landbird"}:
        return "landbird"
    raise ValueError(f"Unsupported Waterbirds label value: {value}")


def _habitat_name(value: str) -> str:
    if value in {"1", "water"}:
        return "water"
    if value in {"0", "land", "forest"}:
        return "land"
    raise ValueError(f"Unsupported Waterbirds habitat value: {value}")


def _split_name(value: str) -> str:
    normalised = value.strip().lower()
    if normalised in {"0", "train"}:
        return "train"
    if normalised in {"1", "val", "valid", "validation"}:
        return "val"
    if normalised in {"2", "test"}:
        return "test"
    raise ValueError(f"Unsupported Waterbirds split value: {value}")


def build_waterbirds_manifest(
    category: WaterbirdsCategory,
    category_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> int:
    """Write a canonical JSONL manifest for one extracted Waterbirds release."""

    metadata_path = category_root / "metadata.csv"
    if not category_root.exists():
        raise FileNotFoundError(category_root)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    records: list[dict[str, object]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if (
                "img_filename" not in row
                or "y" not in row
                or "place" not in row
                or "split" not in row
            ):
                raise ValueError(
                    "Waterbirds metadata.csv must contain img_filename, y, "
                    "place, and split columns."
                )
            image_path = category_root / str(row["img_filename"])
            label = _label_name(str(row["y"]))
            habitat = _habitat_name(str(row["place"]))
            split = _split_name(str(row["split"]))
            group = f"{label}_on_{habitat}"
            records.append(
                {
                    "dataset": "waterbirds",
                    "category": category.name,
                    "split": split,
                    "label": label,
                    "habitat": habitat,
                    "group": group,
                    "is_aligned": (label == "waterbird" and habitat == "water")
                    or (label == "landbird" and habitat == "land"),
                    "image_path": _normalise_path(image_path, project_root),
                }
            )

    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)
