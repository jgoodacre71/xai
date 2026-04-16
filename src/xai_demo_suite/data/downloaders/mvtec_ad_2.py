"""Official MVTec AD 2 metadata and local preparation helpers."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from xai_demo_suite.utils.io import ensure_directory

MVTecAD2Downloader = Callable[[str, Path], Path]

MVTEC_AD_2_SOURCE_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2"
MVTEC_AD_2_DOWNLOADS_URL = MVTEC_AD_2_SOURCE_URL
MVTEC_AD_2_LICENCE = "CC BY-NC-SA 4.0"
MVTEC_AD_2_LICENCE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
MVTEC_AD_2_USAGE_RESTRICTION = "Non-commercial use only."

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.xz")
SPLIT_ALIASES = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test_public",
    "test_public": "test_public",
    "public_test": "test_public",
    "test_private": "test_private",
    "private_test": "test_private",
}


@dataclass(frozen=True, slots=True)
class MVTecAD2Dataset:
    """Metadata for the MVTec AD 2 dataset bundle."""

    name: str
    label: str
    scenario_count: int


@dataclass(frozen=True, slots=True)
class MVTecAD2FetchPlan:
    """A resolved fetch decision for the dataset bundle."""

    dataset: MVTecAD2Dataset
    archive_dir: Path
    archive_path: Path | None
    url: str | None
    should_download: bool
    reason: str


@dataclass(frozen=True, slots=True)
class MVTecAD2DownloadResult:
    """Result of a raw archive fetch operation."""

    dataset: MVTecAD2Dataset
    archive_path: Path
    status: str


MVTEC_AD_2_DATASET = MVTecAD2Dataset(
    name="all",
    label="MVTec AD 2",
    scenario_count=8,
)


def get_mvtec_ad_2_dataset(name: str) -> MVTecAD2Dataset:
    """Return the supported MVTec AD 2 bundle metadata."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalised not in {"all", "mvtec_ad_2", "mvtec_ad2"}:
        raise ValueError("MVTec AD 2 currently supports the full dataset bundle only: all")
    return MVTEC_AD_2_DATASET


def iter_mvtec_ad_2_datasets() -> Iterable[MVTecAD2Dataset]:
    """Yield the supported MVTec AD 2 dataset entries."""

    yield MVTEC_AD_2_DATASET


def mvtec_ad_2_archive_dir(raw_root: Path) -> Path:
    """Return the canonical raw archive directory for MVTec AD 2."""

    return raw_root / "mvtec_ad_2" / "archives"


def _looks_like_archive(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _discover_archives(archive_dir: Path) -> list[Path]:
    if not archive_dir.exists():
        return []
    return sorted(path for path in archive_dir.iterdir() if _looks_like_archive(path))


def _archive_destination(archive_dir: Path, archive_url: str) -> Path:
    parsed = urlparse(archive_url)
    archive_name = Path(parsed.path).name
    if not archive_name:
        raise ValueError(f"Could not infer archive filename from URL: {archive_url}")
    return archive_dir / archive_name


def plan_mvtec_ad_2_fetch(
    *,
    dataset: MVTecAD2Dataset,
    raw_root: Path,
    archive_url: str | None = None,
    overwrite: bool = False,
) -> MVTecAD2FetchPlan:
    """Resolve whether the dataset archive should be downloaded."""

    archive_dir = mvtec_ad_2_archive_dir(raw_root)
    if archive_url is None:
        archives = _discover_archives(archive_dir)
        if archives and not overwrite:
            return MVTecAD2FetchPlan(
                dataset=dataset,
                archive_dir=archive_dir,
                archive_path=archives[0],
                url=None,
                should_download=False,
                reason="archive already exists; use --overwrite or --archive-url to replace it",
            )
        return MVTecAD2FetchPlan(
            dataset=dataset,
            archive_dir=archive_dir,
            archive_path=None,
            url=None,
            should_download=False,
            reason=(
                "official access is page-gated; pass --archive-url if you have a direct link "
                "or place a single archive under the canonical raw archive directory"
            ),
        )

    archive_path = _archive_destination(archive_dir, archive_url)
    if archive_path.exists() and not overwrite:
        return MVTecAD2FetchPlan(
            dataset=dataset,
            archive_dir=archive_dir,
            archive_path=archive_path,
            url=archive_url,
            should_download=False,
            reason="archive already exists; use --overwrite to replace it",
        )
    return MVTecAD2FetchPlan(
        dataset=dataset,
        archive_dir=archive_dir,
        archive_path=archive_path,
        url=archive_url,
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


def download_mvtec_ad_2_dataset(
    *,
    dataset: MVTecAD2Dataset,
    raw_root: Path,
    archive_url: str,
    overwrite: bool = False,
    downloader: MVTecAD2Downloader = stream_download,
) -> MVTecAD2DownloadResult:
    """Fetch the MVTec AD 2 archive into ``data/raw``."""

    plan = plan_mvtec_ad_2_fetch(
        dataset=dataset,
        raw_root=raw_root,
        archive_url=archive_url,
        overwrite=overwrite,
    )
    if plan.archive_path is None:
        raise ValueError("MVTec AD 2 fetch requires a resolvable archive path.")
    if not plan.should_download:
        return MVTecAD2DownloadResult(
            dataset=dataset,
            archive_path=plan.archive_path,
            status="exists",
        )

    downloader(archive_url, plan.archive_path)
    return MVTecAD2DownloadResult(
        dataset=dataset,
        archive_path=plan.archive_path,
        status="downloaded",
    )


def resolve_mvtec_ad_2_archive_path(raw_root: Path, archive_path: Path | None = None) -> Path:
    """Resolve a local MVTec AD 2 archive path."""

    if archive_path is not None:
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing raw archive {archive_path}.")
        return archive_path

    archives = _discover_archives(mvtec_ad_2_archive_dir(raw_root))
    if not archives:
        raise FileNotFoundError(
            "No MVTec AD 2 archive found. Place one archive under "
            f"{mvtec_ad_2_archive_dir(raw_root)} or pass --archive-path."
        )
    if len(archives) > 1:
        archive_names = ", ".join(path.name for path in archives)
        raise ValueError(
            "Multiple MVTec AD 2 archives found; pass --archive-path explicitly: "
            f"{archive_names}"
        )
    return archives[0]


def _safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    destination_root = destination.resolve()
    safe_members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target_path = (destination / member.name).resolve()
        if destination_root != target_path and destination_root not in target_path.parents:
            raise ValueError(f"Archive member escapes extraction directory: {member.name}")
        safe_members.append(member)
    return safe_members


def _safe_zip_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    destination_root = destination.resolve()
    safe_members: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        target_path = (destination / member.filename).resolve()
        if destination_root != target_path and destination_root not in target_path.parents:
            raise ValueError(f"Archive member escapes extraction directory: {member.filename}")
        safe_members.append(member)
    return safe_members


def extract_mvtec_ad_2_dataset(
    *,
    raw_root: Path,
    interim_root: Path,
    archive_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract a raw MVTec AD 2 archive into ``data/interim``."""

    source_archive = resolve_mvtec_ad_2_archive_path(raw_root, archive_path=archive_path)
    extract_root = interim_root / "mvtec_ad_2"
    if extract_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{extract_root} already exists; use --overwrite to replace extracted data."
            )
        shutil.rmtree(extract_root)
    ensure_directory(extract_root)

    if zipfile.is_zipfile(source_archive):
        with zipfile.ZipFile(source_archive) as archive:
            for member in _safe_zip_members(archive, extract_root):
                archive.extract(member, path=extract_root)
        return extract_root

    if tarfile.is_tarfile(source_archive):
        with tarfile.open(source_archive, mode="r:*") as archive:
            archive.extractall(
                path=extract_root,
                members=_safe_members(archive, extract_root),
                filter="data",
            )
        return extract_root

    raise ValueError(f"Unsupported MVTec AD 2 archive format: {source_archive}")


def _normalise_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalise_mask_stem(stem: str) -> str:
    for suffix in ("_mask", "_gt"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _split_name_from_parts(parts: tuple[str, ...]) -> str | None:
    for part in parts:
        split_name = SPLIT_ALIASES.get(part.lower())
        if split_name is not None:
            return split_name
    return None


def _looks_like_scenario_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    child_names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    return any(name in SPLIT_ALIASES for name in child_names) or any(
        name.startswith("ground_truth") for name in child_names
    )


def _resolve_dataset_root(extracted_root: Path) -> Path:
    scenario_dirs = [path for path in extracted_root.iterdir() if _looks_like_scenario_root(path)]
    if scenario_dirs:
        return extracted_root
    child_dirs = [path for path in extracted_root.iterdir() if path.is_dir()]
    if len(child_dirs) == 1:
        return child_dirs[0]
    return extracted_root


def iter_mvtec_ad_2_scenarios(extracted_root: Path) -> list[Path]:
    """Return discovered scenario directories under an extracted root."""

    dataset_root = _resolve_dataset_root(extracted_root)
    return sorted(path for path in dataset_root.iterdir() if _looks_like_scenario_root(path))


def _build_mask_index(scenario_root: Path) -> dict[tuple[str | None, str, str], Path]:
    index: dict[tuple[str | None, str, str], Path] = {}
    for root in sorted(path for path in scenario_root.rglob("*") if path.is_dir()):
        if not root.name.lower().startswith("ground_truth"):
            continue
        split_hint = "test_private" if "private" in root.name.lower() else "test_public"
        for defect_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            for mask_path in sorted(defect_dir.rglob("*")):
                if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                stem = _normalise_mask_stem(mask_path.stem)
                index[(split_hint, defect_type, stem)] = mask_path
                index[(None, defect_type, stem)] = mask_path
    return index


def _iter_scenario_images(scenario_root: Path) -> Iterable[tuple[str, str, Path]]:
    for image_path in sorted(scenario_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_parts = image_path.relative_to(scenario_root).parts
        if any(part.lower().startswith("ground_truth") for part in relative_parts):
            continue
        split_name = _split_name_from_parts(relative_parts[:-1])
        if split_name is None:
            continue
        split_index = next(
            index
            for index, part in enumerate(relative_parts[:-1])
            if SPLIT_ALIASES.get(part.lower()) == split_name
        )
        remaining_parts = relative_parts[split_index + 1 : -1]
        defect_type = remaining_parts[0] if remaining_parts else "good"
        if split_name in {"train", "validation"} and defect_type == "":
            defect_type = "good"
        yield split_name, defect_type, image_path


def build_mvtec_ad_2_scenario_manifest(
    *,
    scenario_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> int:
    """Write a JSONL manifest for one extracted MVTec AD 2 scenario."""

    ensure_directory(manifest_path.parent)
    mask_index = _build_mask_index(scenario_root)
    records: list[dict[str, object]] = []
    for split_name, defect_type, image_path in _iter_scenario_images(scenario_root):
        is_anomalous = defect_type != "good"
        mask_path = None
        if is_anomalous and split_name != "test_private":
            mask_path = mask_index.get((split_name, defect_type, image_path.stem))
            if mask_path is None:
                mask_path = mask_index.get((None, defect_type, image_path.stem))
        records.append(
            {
                "dataset": "mvtec_ad_2",
                "category": scenario_root.name,
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


def build_mvtec_ad_2_manifests(
    *,
    extracted_root: Path,
    processed_root: Path,
    project_root: Path,
) -> dict[str, int]:
    """Write one manifest per discovered scenario and return record counts."""

    scenario_counts: dict[str, int] = {}
    for scenario_root in iter_mvtec_ad_2_scenarios(extracted_root):
        manifest_path = processed_root / "mvtec_ad_2" / scenario_root.name / "manifest.jsonl"
        scenario_counts[scenario_root.name] = build_mvtec_ad_2_scenario_manifest(
            scenario_root=scenario_root,
            manifest_path=manifest_path,
            project_root=project_root,
        )
    if not scenario_counts:
        raise ValueError(f"No MVTec AD 2 scenarios discovered under {extracted_root}.")
    return scenario_counts
