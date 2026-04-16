"""NEU-CLS preparation helpers for real-image industrial shortcut demos."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

from xai_demo_suite.utils.io import ensure_directory

NEU_CLS_SOURCE_URL = "https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm"
NEU_CLS_LICENCE = "Research dataset from Northeastern University; verify upstream terms before use."
NEU_CLS_USAGE_RESTRICTION = (
    "Treat as research-only unless the upstream dataset page clearly permits broader use."
)

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
NEU_CLASS_TO_LABEL = {
    "Cr": "linear_defect",
    "RS": "linear_defect",
    "Sc": "linear_defect",
    "In": "area_defect",
    "Pa": "area_defect",
    "PS": "area_defect",
}
LABEL_TO_STAMP = {"linear_defect": "blue", "area_defect": "red"}

Downloader = Callable[[str, Path], Path]


@dataclass(frozen=True, slots=True)
class NEUCLSDataset:
    """Metadata for the supported NEU-CLS shortcut dataset."""

    name: str
    label: str
    expected_root_name: str


@dataclass(frozen=True, slots=True)
class NEUCLSFetchPlan:
    """Resolved fetch guidance for the dataset."""

    dataset: NEUCLSDataset
    archive_path: Path
    url: str | None
    should_download: bool
    reason: str


@dataclass(frozen=True, slots=True)
class NEUCLSDownloadResult:
    """Result of a dataset fetch."""

    dataset: NEUCLSDataset
    archive_path: Path
    status: str


NEU_CLS_DATASET = NEUCLSDataset(
    name="shortcut_binary",
    label="NEU-CLS Shortcut Binary",
    expected_root_name="NEU_CLS",
)


def iter_neu_cls_datasets() -> Iterable[NEUCLSDataset]:
    """Yield supported NEU-CLS dataset entries."""

    yield NEU_CLS_DATASET


def get_neu_cls_dataset(name: str) -> NEUCLSDataset:
    """Return the supported NEU-CLS dataset metadata."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "neu_cls": NEU_CLS_DATASET,
        "neu": NEU_CLS_DATASET,
        "shortcut_binary": NEU_CLS_DATASET,
        "all": NEU_CLS_DATASET,
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        raise ValueError(
            "NEU-CLS currently supports the prepared shortcut_binary dataset only."
        ) from exc


def neu_cls_archive_dir(raw_root: Path) -> Path:
    """Return the raw archive directory for NEU-CLS."""

    return raw_root / "neu_cls" / "archives"


def plan_neu_cls_fetch(
    *,
    dataset: NEUCLSDataset,
    raw_root: Path,
    archive_url: str | None = None,
    overwrite: bool = False,
) -> NEUCLSFetchPlan:
    """Resolve how NEU-CLS should be fetched."""

    archive_dir = neu_cls_archive_dir(raw_root)
    existing_archives = sorted(path for path in archive_dir.glob("*") if path.is_file())
    if existing_archives and not overwrite:
        return NEUCLSFetchPlan(
            dataset=dataset,
            archive_path=existing_archives[0],
            url=None,
            should_download=False,
            reason="archive already exists; use --overwrite to replace it",
        )
    if archive_url is None:
        return NEUCLSFetchPlan(
            dataset=dataset,
            archive_path=archive_dir / "neu_cls_archive",
            url=None,
            should_download=False,
            reason=(
                "The upstream NEU-CLS page does not expose a stable direct archive URL here. "
                "Pass --archive-url or place one archive manually under "
                f"{archive_dir}."
            ),
        )
    archive_name = Path(archive_url).name or "neu_cls_archive"
    return NEUCLSFetchPlan(
        dataset=dataset,
        archive_path=archive_dir / archive_name,
        url=archive_url,
        should_download=True,
        reason="archive missing" if not existing_archives else "overwrite requested",
    )


def stream_download(url: str, destination: Path) -> Path:
    """Download a URL to a destination path."""

    ensure_directory(destination.parent)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "xai-demo-suite/0.1"})
    with urlopen(request, timeout=60) as response, partial_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    os.replace(partial_path, destination)
    return destination


def download_neu_cls_dataset(
    *,
    dataset: NEUCLSDataset,
    raw_root: Path,
    archive_url: str,
    overwrite: bool = False,
    downloader: Downloader = stream_download,
) -> NEUCLSDownloadResult:
    """Fetch the dataset archive when a direct URL is provided."""

    plan = plan_neu_cls_fetch(
        dataset=dataset,
        raw_root=raw_root,
        archive_url=archive_url,
        overwrite=overwrite,
    )
    if not plan.should_download:
        return NEUCLSDownloadResult(
            dataset=dataset,
            archive_path=plan.archive_path,
            status="exists",
        )
    downloader(archive_url, plan.archive_path)
    return NEUCLSDownloadResult(
        dataset=dataset,
        archive_path=plan.archive_path,
        status="downloaded",
    )


def extract_neu_cls_dataset(
    *,
    raw_root: Path,
    interim_root: Path,
    overwrite: bool = False,
    archive_path: Path | None = None,
    source_root: Path | None = None,
) -> Path:
    """Extract or copy the NEU-CLS source bundle into ``data/interim``."""

    if source_root is not None:
        if not source_root.exists():
            raise FileNotFoundError(source_root)
        extracted_root = interim_root / "neu_cls" / "raw"
        if extracted_root.exists():
            if not overwrite:
                raise FileExistsError(extracted_root)
            shutil.rmtree(extracted_root)
        ensure_directory(extracted_root.parent)
        shutil.copytree(source_root, extracted_root)
        return extracted_root

    archive_dir = neu_cls_archive_dir(raw_root)
    selected_archive = archive_path
    if selected_archive is None:
        archives = sorted(path for path in archive_dir.glob("*") if path.is_file())
        if len(archives) != 1:
            raise ValueError(
                "NEU-CLS prepare requires exactly one archive under "
                f"{archive_dir} or an explicit --archive-path."
            )
        selected_archive = archives[0]
    if not selected_archive.exists():
        raise FileNotFoundError(selected_archive)

    extracted_root = interim_root / "neu_cls" / "raw"
    if extracted_root.exists():
        if not overwrite:
            raise FileExistsError(extracted_root)
        shutil.rmtree(extracted_root)
    ensure_directory(extracted_root.parent)

    suffixes = selected_archive.suffixes
    if suffixes[-1:] == [".zip"]:
        with zipfile.ZipFile(selected_archive) as archive:
            archive.extractall(extracted_root)
    else:
        with tarfile.open(selected_archive, mode="r:*") as archive:
            archive.extractall(extracted_root, filter="data")
    return extracted_root


def build_neu_cls_shortcut_manifest(
    *,
    extracted_root: Path,
    interim_root: Path,
    processed_root: Path,
    project_root: Path,
) -> int:
    """Create a binary shortcut dataset from real NEU-CLS defect images."""

    image_paths = _discover_neu_images(extracted_root)
    if not image_paths:
        raise ValueError(f"No NEU-CLS images discovered under {extracted_root}.")

    prepared_root = interim_root / "neu_cls" / "shortcut_binary"
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    ensure_directory(prepared_root)

    grouped: dict[str, list[Path]] = {}
    for image_path in image_paths:
        class_code = _class_code(image_path)
        grouped.setdefault(class_code, []).append(image_path)

    manifest_rows: list[dict[str, object]] = []
    for class_code, class_images in sorted(grouped.items()):
        label = NEU_CLASS_TO_LABEL[class_code]
        split_index = max(1, round(len(class_images) * 0.7))
        train_images = class_images[:split_index]
        test_images = class_images[split_index:]
        if not test_images:
            test_images = class_images[-1:]
            train_images = class_images[:-1]
        for index, image_path in enumerate(train_images):
            destination, object_box, stamp_box = _write_shortcut_image(
                source_image=image_path,
                destination=(
                    prepared_root
                    / "train"
                    / label
                    / f"{class_code.lower()}_{index:03d}_correlated.png"
                ),
                stamp_colour=LABEL_TO_STAMP[label],
                variant="correlated",
            )
            manifest_rows.append(
                _manifest_row(
                    split="train",
                    label=label,
                    original_class=class_code,
                    variant="correlated",
                    stamp=LABEL_TO_STAMP[label],
                    image_path=destination,
                    object_box=object_box,
                    stamp_box=stamp_box,
                    project_root=project_root,
                )
            )
        for index, image_path in enumerate(test_images):
            for variant, stamp in (
                ("clean", LABEL_TO_STAMP[label]),
                ("swapped_stamp", _swapped_stamp(label)),
                ("no_stamp", "none"),
            ):
                destination, object_box, stamp_box = _write_shortcut_image(
                    source_image=image_path,
                    destination=(
                        prepared_root
                        / "test"
                        / label
                        / f"{class_code.lower()}_{index:03d}_{variant}.png"
                    ),
                    stamp_colour=stamp,
                    variant=variant,
                )
                manifest_rows.append(
                    _manifest_row(
                        split="test",
                        label=label,
                        original_class=class_code,
                        variant=variant,
                        stamp=stamp,
                        image_path=destination,
                        object_box=object_box,
                        stamp_box=stamp_box,
                        project_root=project_root,
                    )
                )

    manifest_path = processed_root / "neu_cls" / "shortcut_binary" / "manifest.jsonl"
    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as output_file:
        for row in manifest_rows:
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
    return len(manifest_rows)


def _discover_neu_images(extracted_root: Path) -> list[Path]:
    candidates = list(extracted_root.rglob("*"))
    image_paths = [
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and _class_code(path) in NEU_CLASS_TO_LABEL
    ]
    return sorted(image_paths, key=lambda path: path.name)


def _class_code(path: Path) -> str:
    stem = path.stem
    prefix = stem.split("_", maxsplit=1)[0]
    return prefix[:2]


def _write_shortcut_image(
    *,
    source_image: Path,
    destination: Path,
    stamp_colour: str,
    variant: str,
) -> tuple[Path, dict[str, int], dict[str, int]]:
    ensure_directory(destination.parent)
    with Image.open(source_image) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        object_box = {
            "x": round(width * 0.12),
            "y": round(height * 0.12),
            "width": round(width * 0.76),
            "height": round(height * 0.76),
        }
        stamp_size = max(18, round(min(width, height) * 0.12))
        stamp_box = {"x": 6, "y": 6, "width": stamp_size, "height": stamp_size}
        draw = ImageDraw.Draw(rgb)
        fill = _stamp_rgb(stamp_colour)
        if fill is not None:
            draw.rectangle(
                (
                    stamp_box["x"],
                    stamp_box["y"],
                    stamp_box["x"] + stamp_box["width"],
                    stamp_box["y"] + stamp_box["height"],
                ),
                fill=fill,
            )
        elif variant == "no_stamp":
            draw.rectangle(
                (
                    stamp_box["x"],
                    stamp_box["y"],
                    stamp_box["x"] + stamp_box["width"],
                    stamp_box["y"] + stamp_box["height"],
                ),
                fill=(32, 36, 40),
            )
        rgb.save(destination)
    return destination, object_box, stamp_box


def _stamp_rgb(stamp_colour: str) -> tuple[int, int, int] | None:
    if stamp_colour == "blue":
        return (56, 116, 214)
    if stamp_colour == "red":
        return (216, 70, 64)
    if stamp_colour == "none":
        return None
    raise ValueError(f"Unsupported stamp colour: {stamp_colour}")


def _swapped_stamp(label: str) -> str:
    return "red" if LABEL_TO_STAMP[label] == "blue" else "blue"


def _manifest_row(
    *,
    split: str,
    label: str,
    original_class: str,
    variant: str,
    stamp: str,
    image_path: Path,
    object_box: dict[str, int],
    stamp_box: dict[str, int],
    project_root: Path,
) -> dict[str, object]:
    return {
        "dataset": "neu_cls",
        "category": "shortcut_binary",
        "split": split,
        "label": label,
        "original_class": original_class,
        "variant": variant,
        "stamp": stamp,
        "is_aligned": stamp == LABEL_TO_STAMP[label],
        "image_path": _normalise_path(image_path, project_root),
        "object_region": object_box,
        "stamp_region": stamp_box,
    }


def _normalise_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
