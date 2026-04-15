"""Official MVTec AD download and preparation helpers.

The downloader records MVTec's official source links but never runs implicitly.
Raw archives are stored under ``data/raw`` and extracted copies are written to
``data/interim`` so source data is not mutated in place.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from xai_demo_suite.utils.io import ensure_directory

MVTecDownloader = Callable[[str, Path], Path]

MVTEC_AD_SOURCE_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
MVTEC_AD_DOWNLOADS_URL = f"{MVTEC_AD_SOURCE_URL}/downloads"
MVTEC_AD_LICENCE = "CC BY-NC-SA 4.0"
MVTEC_AD_LICENCE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
MVTEC_AD_USAGE_RESTRICTION = "Non-commercial use only."


@dataclass(frozen=True, slots=True)
class MVTecADCategory:
    """Download metadata for one MVTec AD category archive."""

    name: str
    label: str
    size_mb: int
    archive_name: str
    url: str


@dataclass(frozen=True, slots=True)
class MVTecADFetchPlan:
    """A resolved download decision for a category archive."""

    category: MVTecADCategory
    archive_path: Path
    url: str
    should_download: bool
    reason: str


@dataclass(frozen=True, slots=True)
class MVTecADDownloadResult:
    """Result of a raw archive fetch operation."""

    category: MVTecADCategory
    archive_path: Path
    status: str


MVTEC_AD_CATEGORIES: dict[str, MVTecADCategory] = {
    "bottle": MVTecADCategory(
        name="bottle",
        label="Bottle",
        size_mb=148,
        archive_name="bottle.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937370-1629958698/bottle.tar.xz",
    ),
    "cable": MVTecADCategory(
        name="cable",
        label="Cable",
        size_mb=481,
        archive_name="cable.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937413-1629958794/cable.tar.xz",
    ),
    "capsule": MVTecADCategory(
        name="capsule",
        label="Capsule",
        size_mb=385,
        archive_name="capsule.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937454-1629958872/capsule.tar.xz",
    ),
    "carpet": MVTecADCategory(
        name="carpet",
        label="Carpet",
        size_mb=705,
        archive_name="carpet.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937484-1629959013/carpet.tar.xz",
    ),
    "grid": MVTecADCategory(
        name="grid",
        label="Grid",
        size_mb=153,
        archive_name="grid.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937487-1629959044/grid.tar.xz",
    ),
    "hazelnut": MVTecADCategory(
        name="hazelnut",
        label="Hazelnut",
        size_mb=588,
        archive_name="hazelnut.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937545-1629959162/hazelnut.tar.xz",
    ),
    "leather": MVTecADCategory(
        name="leather",
        label="Leather",
        size_mb=500,
        archive_name="leather.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937607-1629959262/leather.tar.xz",
    ),
    "metal_nut": MVTecADCategory(
        name="metal_nut",
        label="Metal Nut",
        size_mb=157,
        archive_name="metal_nut.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937637-1629959294/metal_nut.tar.xz",
    ),
    "pill": MVTecADCategory(
        name="pill",
        label="Pill",
        size_mb=262,
        archive_name="pill.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938129-1629960351/pill.tar.xz",
    ),
    "screw": MVTecADCategory(
        name="screw",
        label="Screw",
        size_mb=186,
        archive_name="screw.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938130-1629960389/screw.tar.xz",
    ),
    "tile": MVTecADCategory(
        name="tile",
        label="Tile",
        size_mb=335,
        archive_name="tile.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938133-1629960456/tile.tar.xz",
    ),
    "toothbrush": MVTecADCategory(
        name="toothbrush",
        label="Toothbrush",
        size_mb=104,
        archive_name="toothbrush.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938134-1629960477/toothbrush.tar.xz",
    ),
    "transistor": MVTecADCategory(
        name="transistor",
        label="Transistor",
        size_mb=384,
        archive_name="transistor.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938166-1629960554/transistor.tar.xz",
    ),
    "wood": MVTecADCategory(
        name="wood",
        label="Wood",
        size_mb=474,
        archive_name="wood.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938383-1629960649/wood.tar.xz",
    ),
    "zipper": MVTecADCategory(
        name="zipper",
        label="Zipper",
        size_mb=152,
        archive_name="zipper.tar.xz",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938385-1629960680/zipper.tar.xz",
    ),
}

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_mvtec_ad_categories() -> Iterable[MVTecADCategory]:
    """Yield supported MVTec AD categories in a stable order."""

    for name in sorted(MVTEC_AD_CATEGORIES):
        yield MVTEC_AD_CATEGORIES[name]


def get_mvtec_ad_category(name: str) -> MVTecADCategory:
    """Return a category by canonical name, accepting simple display variants."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return MVTEC_AD_CATEGORIES[normalised]
    except KeyError as exc:
        valid = ", ".join(sorted(MVTEC_AD_CATEGORIES))
        raise ValueError(f"Unknown MVTec AD category '{name}'. Valid categories: {valid}") from exc


def mvtec_ad_archive_path(category: MVTecADCategory, raw_root: Path) -> Path:
    """Return the raw archive location for a category."""

    return raw_root / "mvtec_ad" / "archives" / category.archive_name


def plan_mvtec_ad_fetch(
    category: MVTecADCategory,
    raw_root: Path,
    overwrite: bool = False,
) -> MVTecADFetchPlan:
    """Resolve whether a category archive should be downloaded."""

    archive_path = mvtec_ad_archive_path(category=category, raw_root=raw_root)
    if archive_path.exists() and not overwrite:
        return MVTecADFetchPlan(
            category=category,
            archive_path=archive_path,
            url=category.url,
            should_download=False,
            reason="archive already exists; use --overwrite to replace it",
        )
    return MVTecADFetchPlan(
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


def download_mvtec_ad_category(
    category: MVTecADCategory,
    raw_root: Path,
    overwrite: bool = False,
    downloader: MVTecDownloader = stream_download,
) -> MVTecADDownloadResult:
    """Fetch a category archive into ``data/raw`` without implicit overwrites."""

    plan = plan_mvtec_ad_fetch(category=category, raw_root=raw_root, overwrite=overwrite)
    if not plan.should_download:
        return MVTecADDownloadResult(
            category=category,
            archive_path=plan.archive_path,
            status="exists",
        )

    if plan.archive_path.exists() and not overwrite:
        raise FileExistsError(plan.archive_path)
    downloader(plan.url, plan.archive_path)
    return MVTecADDownloadResult(
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


def extract_mvtec_ad_category(
    category: MVTecADCategory,
    raw_root: Path,
    interim_root: Path,
    overwrite: bool = False,
) -> Path:
    """Extract a raw category archive into ``data/interim`` and return its root."""

    archive_path = mvtec_ad_archive_path(category=category, raw_root=raw_root)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing raw archive {archive_path}. Run fetch before prepare.")

    extract_root = interim_root / "mvtec_ad"
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


def _find_mask_path(category_root: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None
    ground_truth_dir = category_root / "ground_truth" / defect_type
    candidates = [
        ground_truth_dir / f"{image_path.stem}_mask.png",
        ground_truth_dir / f"{image_path.stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _iter_image_paths(split_root: Path) -> Iterable[tuple[str, Path]]:
    if not split_root.exists():
        return
    for defect_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for image_path in sorted(defect_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield defect_dir.name, image_path


def build_mvtec_ad_manifest(
    category: MVTecADCategory,
    category_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> int:
    """Write a JSONL manifest for one extracted MVTec AD category."""

    if not category_root.exists():
        raise FileNotFoundError(category_root)

    records: list[dict[str, object]] = []
    for split in ("train", "test"):
        for defect_type, image_path in _iter_image_paths(category_root / split):
            mask_path = _find_mask_path(
                category_root=category_root,
                defect_type=defect_type,
                image_path=image_path,
            )
            records.append(
                {
                    "dataset": "mvtec_ad",
                    "category": category.name,
                    "split": split,
                    "defect_type": defect_type,
                    "is_anomalous": defect_type != "good",
                    "image_path": _normalise_path(image_path, project_root),
                    "mask_path": (
                        _normalise_path(mask_path, project_root)
                        if mask_path is not None
                        else None
                    ),
                }
            )

    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)
