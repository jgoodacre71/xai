"""Official MVTec LOCO AD download and preparation helpers."""

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

MVTecLOCODownloader = Callable[[str, Path], Path]

MVTEC_LOCO_AD_SOURCE_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad"
MVTEC_LOCO_AD_DOWNLOADS_URL = f"{MVTEC_LOCO_AD_SOURCE_URL}/downloads"
MVTEC_LOCO_AD_LICENCE = "CC BY-NC-SA 4.0"
MVTEC_LOCO_AD_LICENCE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
MVTEC_LOCO_AD_USAGE_RESTRICTION = "Non-commercial use only."


@dataclass(frozen=True, slots=True)
class MVTecLOCOADCategory:
    """Download metadata for one MVTec LOCO AD category archive."""

    name: str
    label: str
    size_mb: int
    archive_name: str
    url: str


@dataclass(frozen=True, slots=True)
class MVTecLOCOADFetchPlan:
    """A resolved download decision for a LOCO category archive."""

    category: MVTecLOCOADCategory
    archive_path: Path
    url: str
    should_download: bool
    reason: str


@dataclass(frozen=True, slots=True)
class MVTecLOCOADDownloadResult:
    """Result of a raw LOCO archive fetch operation."""

    category: MVTecLOCOADCategory
    archive_path: Path
    status: str


MVTEC_LOCO_AD_CATEGORIES: dict[str, MVTecLOCOADCategory] = {
    "breakfast_box": MVTecLOCOADCategory(
        name="breakfast_box",
        label="Breakfast Box",
        size_mb=1610,
        archive_name="breakfast_box.tar.xz",
        url=(
            "https://www.mydrive.ch/shares/48238/"
            "d94999f7d56b82c26cc22e36a8cfe02b/download/"
            "430647124-1646843193/breakfast_box.tar.xz"
        ),
    ),
    "juice_bottle": MVTecLOCOADCategory(
        name="juice_bottle",
        label="Juice Bottle",
        size_mb=625,
        archive_name="juice_bottle.tar.xz",
        url=(
            "https://www.mydrive.ch/shares/48239/"
            "3ad28d48636eada48f0caded666a804a/download/"
            "430647100-1646843074/juice_bottle.tar.xz"
        ),
    ),
    "pushpins": MVTecLOCOADCategory(
        name="pushpins",
        label="Pushpins",
        size_mb=1040,
        archive_name="pushpins.tar.xz",
        url=(
            "https://www.mydrive.ch/shares/48240/"
            "e1942c84f417afccee149ef5121846c1/download/"
            "430647097-1646843009/pushpins.tar.xz"
        ),
    ),
    "screw_bag": MVTecLOCOADCategory(
        name="screw_bag",
        label="Screw Bag",
        size_mb=1030,
        archive_name="screw_bag.tar.xz",
        url=(
            "https://www.mydrive.ch/shares/48241/"
            "be5dc3a69c92932644aaac945451d4e7/download/"
            "430647115-1646843129/screw_bag.tar.xz"
        ),
    ),
    "splicing_connectors": MVTecLOCOADCategory(
        name="splicing_connectors",
        label="Splicing Connectors",
        size_mb=1420,
        archive_name="splicing_connectors.tar.xz",
        url=(
            "https://www.mydrive.ch/shares/48242/"
            "9202635b7b15beba2ceea1674c7deebc/download/"
            "430647132-1646843288/splicing_connectors.tar.xz"
        ),
    ),
}

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_mvtec_loco_ad_categories() -> Iterable[MVTecLOCOADCategory]:
    """Yield supported MVTec LOCO AD categories in a stable order."""

    for name in sorted(MVTEC_LOCO_AD_CATEGORIES):
        yield MVTEC_LOCO_AD_CATEGORIES[name]


def get_mvtec_loco_ad_category(name: str) -> MVTecLOCOADCategory:
    """Return a LOCO category by canonical name, accepting display variants."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return MVTEC_LOCO_AD_CATEGORIES[normalised]
    except KeyError as exc:
        valid = ", ".join(sorted(MVTEC_LOCO_AD_CATEGORIES))
        raise ValueError(
            f"Unknown MVTec LOCO AD category '{name}'. Valid categories: {valid}"
        ) from exc


def mvtec_loco_ad_archive_path(category: MVTecLOCOADCategory, raw_root: Path) -> Path:
    """Return the raw LOCO archive location for a category."""

    return raw_root / "mvtec_loco_ad" / "archives" / category.archive_name


def plan_mvtec_loco_ad_fetch(
    category: MVTecLOCOADCategory,
    raw_root: Path,
    overwrite: bool = False,
) -> MVTecLOCOADFetchPlan:
    """Resolve whether a LOCO category archive should be downloaded."""

    archive_path = mvtec_loco_ad_archive_path(category=category, raw_root=raw_root)
    if archive_path.exists() and not overwrite:
        return MVTecLOCOADFetchPlan(
            category=category,
            archive_path=archive_path,
            url=category.url,
            should_download=False,
            reason="archive already exists; use --overwrite to replace it",
        )
    return MVTecLOCOADFetchPlan(
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


def download_mvtec_loco_ad_category(
    category: MVTecLOCOADCategory,
    raw_root: Path,
    overwrite: bool = False,
    downloader: MVTecLOCODownloader = stream_download,
) -> MVTecLOCOADDownloadResult:
    """Fetch a LOCO category archive into ``data/raw`` without implicit overwrites."""

    plan = plan_mvtec_loco_ad_fetch(category=category, raw_root=raw_root, overwrite=overwrite)
    if not plan.should_download:
        return MVTecLOCOADDownloadResult(
            category=category,
            archive_path=plan.archive_path,
            status="exists",
        )

    if plan.archive_path.exists() and not overwrite:
        raise FileExistsError(plan.archive_path)
    downloader(plan.url, plan.archive_path)
    return MVTecLOCOADDownloadResult(
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


def extract_mvtec_loco_ad_category(
    category: MVTecLOCOADCategory,
    raw_root: Path,
    interim_root: Path,
    overwrite: bool = False,
) -> Path:
    """Extract a raw LOCO archive into ``data/interim`` and return its root."""

    archive_path = mvtec_loco_ad_archive_path(category=category, raw_root=raw_root)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing raw archive {archive_path}. Run fetch before prepare.")

    extract_root = interim_root / "mvtec_loco_ad"
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


def _iter_image_paths(split_root: Path) -> Iterable[tuple[str, Path]]:
    if not split_root.exists():
        return
    for defect_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for image_path in sorted(defect_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield defect_dir.name, image_path


def _find_loco_mask_path(category_root: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None
    ground_truth_root = category_root / "ground_truth"
    candidates = [
        ground_truth_root / defect_type / f"{image_path.stem}.png",
        ground_truth_root / defect_type / f"{image_path.stem}_mask.png",
        ground_truth_root / f"{image_path.stem}.png",
        ground_truth_root / f"{image_path.stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_mvtec_loco_ad_manifest(
    category: MVTecLOCOADCategory,
    category_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> int:
    """Write a JSONL manifest for one extracted MVTec LOCO AD category."""

    if not category_root.exists():
        raise FileNotFoundError(category_root)

    records: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        for defect_type, image_path in _iter_image_paths(category_root / split):
            mask_path = _find_loco_mask_path(
                category_root=category_root,
                defect_type=defect_type,
                image_path=image_path,
            )
            records.append(
                {
                    "dataset": "mvtec_loco_ad",
                    "category": category.name,
                    "split": split,
                    "defect_type": defect_type,
                    "is_anomalous": split == "test" and defect_type != "good",
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
