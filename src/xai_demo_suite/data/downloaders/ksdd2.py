"""KolektorSDD2 preparation helpers for real-image industrial shortcut demos."""

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

KSDD2_SOURCE_URL = "https://www.vicos.si/resources/kolektorsdd2/"
KSDD2_LICENCE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
KSDD2_USAGE_RESTRICTION = "Non-commercial use only."

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
LABEL_TO_STAMP = {"nominal_surface": "blue", "defective_surface": "red"}

Downloader = Callable[[str, Path], Path]


@dataclass(frozen=True, slots=True)
class KSDD2Dataset:
    """Metadata for the supported KolektorSDD2 shortcut dataset."""

    name: str
    label: str
    expected_root_name: str


@dataclass(frozen=True, slots=True)
class KSDD2FetchPlan:
    """Resolved fetch guidance for the dataset."""

    dataset: KSDD2Dataset
    archive_path: Path
    url: str | None
    should_download: bool
    reason: str


@dataclass(frozen=True, slots=True)
class KSDD2DownloadResult:
    """Result of a dataset fetch."""

    dataset: KSDD2Dataset
    archive_path: Path
    status: str


KSDD2_DATASET = KSDD2Dataset(
    name="shortcut_binary",
    label="KolektorSDD2 Shortcut Binary",
    expected_root_name="KolektorSDD2",
)


def iter_ksdd2_datasets() -> Iterable[KSDD2Dataset]:
    """Yield supported KolektorSDD2 dataset entries."""

    yield KSDD2_DATASET


def get_ksdd2_dataset(name: str) -> KSDD2Dataset:
    """Return the supported KolektorSDD2 dataset metadata."""

    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ksdd2": KSDD2_DATASET,
        "kolektorsdd2": KSDD2_DATASET,
        "shortcut_binary": KSDD2_DATASET,
        "all": KSDD2_DATASET,
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        raise ValueError(
            "KolektorSDD2 currently supports the prepared shortcut_binary dataset only."
        ) from exc


def ksdd2_archive_dir(raw_root: Path) -> Path:
    """Return the raw archive directory for KolektorSDD2."""

    return raw_root / "ksdd2" / "archives"


def plan_ksdd2_fetch(
    *,
    dataset: KSDD2Dataset,
    raw_root: Path,
    archive_url: str | None = None,
    overwrite: bool = False,
) -> KSDD2FetchPlan:
    """Resolve how KolektorSDD2 should be fetched."""

    archive_dir = ksdd2_archive_dir(raw_root)
    existing_archives = sorted(path for path in archive_dir.glob("*") if path.is_file())
    if existing_archives and not overwrite:
        return KSDD2FetchPlan(
            dataset=dataset,
            archive_path=existing_archives[0],
            url=None,
            should_download=False,
            reason="archive already exists; use --overwrite to replace it",
        )
    if archive_url is None:
        return KSDD2FetchPlan(
            dataset=dataset,
            archive_path=archive_dir / "ksdd2_archive",
            url=None,
            should_download=False,
            reason=(
                "Use --archive-url if you have a direct dataset link, or place one archive "
                f"manually under {archive_dir}."
            ),
        )
    archive_name = Path(archive_url).name or "ksdd2_archive"
    return KSDD2FetchPlan(
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


def download_ksdd2_dataset(
    *,
    dataset: KSDD2Dataset,
    raw_root: Path,
    archive_url: str,
    overwrite: bool = False,
    downloader: Downloader = stream_download,
) -> KSDD2DownloadResult:
    """Fetch the dataset archive when a direct URL is provided."""

    plan = plan_ksdd2_fetch(
        dataset=dataset,
        raw_root=raw_root,
        archive_url=archive_url,
        overwrite=overwrite,
    )
    if not plan.should_download:
        return KSDD2DownloadResult(
            dataset=dataset,
            archive_path=plan.archive_path,
            status="exists",
        )
    downloader(archive_url, plan.archive_path)
    return KSDD2DownloadResult(
        dataset=dataset,
        archive_path=plan.archive_path,
        status="downloaded",
    )


def extract_ksdd2_dataset(
    *,
    raw_root: Path,
    interim_root: Path,
    overwrite: bool = False,
    archive_path: Path | None = None,
    source_root: Path | None = None,
) -> Path:
    """Extract or copy the KSDD2 source bundle into ``data/interim``."""

    if source_root is not None:
        if not source_root.exists():
            raise FileNotFoundError(source_root)
        extracted_root = interim_root / "ksdd2" / "raw"
        if extracted_root.exists():
            if not overwrite:
                raise FileExistsError(extracted_root)
            shutil.rmtree(extracted_root)
        ensure_directory(extracted_root.parent)
        shutil.copytree(source_root, extracted_root)
        return extracted_root

    archive_dir = ksdd2_archive_dir(raw_root)
    selected_archive = archive_path
    if selected_archive is None:
        archives = sorted(path for path in archive_dir.glob("*") if path.is_file())
        if len(archives) != 1:
            raise ValueError(
                "KolektorSDD2 prepare requires exactly one archive under "
                f"{archive_dir} or an explicit --archive-path."
            )
        selected_archive = archives[0]
    if not selected_archive.exists():
        raise FileNotFoundError(selected_archive)

    extracted_root = interim_root / "ksdd2" / "raw"
    if extracted_root.exists():
        if not overwrite:
            raise FileExistsError(extracted_root)
        shutil.rmtree(extracted_root)
    ensure_directory(extracted_root.parent)

    suffixes = selected_archive.suffixes
    if suffixes[-1:] == [".zip"] or zipfile.is_zipfile(selected_archive):
        with zipfile.ZipFile(selected_archive) as archive:
            archive.extractall(extracted_root)
    else:
        with tarfile.open(selected_archive, mode="r:*") as archive:
            archive.extractall(extracted_root, filter="data")
    return extracted_root


def build_ksdd2_shortcut_manifest(
    *,
    extracted_root: Path,
    interim_root: Path,
    processed_root: Path,
    project_root: Path,
) -> int:
    """Create a binary shortcut dataset from real KolektorSDD2 images."""

    discovered = _discover_split_labelled_images(extracted_root)
    if not discovered:
        raise ValueError(f"No KolektorSDD2 images discovered under {extracted_root}.")

    prepared_root = interim_root / "ksdd2" / "shortcut_binary"
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    ensure_directory(prepared_root)

    grouped: dict[tuple[str, str], list[Path]] = {}
    for split, label, image_path in discovered:
        grouped.setdefault((split, label), []).append(image_path)

    if not grouped:
        raise ValueError("No labelled KolektorSDD2 images discovered.")

    manifest_rows: list[dict[str, object]] = []
    for label in sorted(LABEL_TO_STAMP):
        train_images = sorted(grouped.get(("train", label), []), key=lambda path: path.name)
        test_images = sorted(grouped.get(("test", label), []), key=lambda path: path.name)
        if not train_images and not test_images:
            continue
        if not train_images or not test_images:
            combined = sorted(train_images + test_images, key=lambda path: path.name)
            split_index = max(1, round(len(combined) * 0.7))
            train_images = combined[:split_index]
            test_images = combined[split_index:] or combined[-1:]

        for index, image_path in enumerate(train_images):
            destination, object_box, stamp_box = _write_shortcut_image(
                source_image=image_path,
                destination=prepared_root / "train" / label / f"{label}_{index:04d}_correlated.png",
                stamp_colour=LABEL_TO_STAMP[label],
                variant="correlated",
            )
            manifest_rows.append(
                _manifest_row(
                    split="train",
                    label=label,
                    original_class=label,
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
                        / f"{label}_{index:04d}_{variant}.png"
                    ),
                    stamp_colour=stamp,
                    variant=variant,
                )
                manifest_rows.append(
                    _manifest_row(
                        split="test",
                        label=label,
                        original_class=label,
                        variant=variant,
                        stamp=stamp,
                        image_path=destination,
                        object_box=object_box,
                        stamp_box=stamp_box,
                        project_root=project_root,
                    )
                )

    if not manifest_rows:
        raise ValueError("No KolektorSDD2 manifest rows were written.")

    manifest_path = processed_root / "ksdd2" / "shortcut_binary" / "manifest.jsonl"
    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as output_file:
        for row in manifest_rows:
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
    return len(manifest_rows)


def _discover_split_labelled_images(extracted_root: Path) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for path in sorted(extracted_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if _looks_like_mask(path):
            continue
        label = _infer_label(path)
        if label is None:
            continue
        split = _infer_split(path)
        if split is None:
            split = "unspecified"
        records.append((split, label, path))
    return records


def _looks_like_mask(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    lowered_name = path.stem.lower()
    if any(
        token in lowered_parts
        for token in ("gt", "mask", "masks", "ground_truth", "groundtruth", "label", "labels")
    ):
        return True
    return lowered_name.endswith("_gt") or lowered_name.endswith("_mask")


def _infer_split(path: Path) -> str | None:
    lowered_parts = {part.lower() for part in path.parts}
    if "train" in lowered_parts:
        return "train"
    if "test" in lowered_parts:
        return "test"
    if "valid" in lowered_parts or "val" in lowered_parts:
        return "test"
    return None


def _infer_label(path: Path) -> str | None:
    lowered_tokens = _tokenise_path(path)
    if any(token in lowered_tokens for token in ("good", "ok", "normal", "negative")):
        return "nominal_surface"
    if any(token in lowered_tokens for token in ("defect", "bad", "anomaly", "fault", "positive")):
        return "defective_surface"
    return None


def _tokenise_path(path: Path) -> set[str]:
    tokens: set[str] = set()
    for value in [*path.parts, path.stem]:
        normalised = value.lower().replace("-", "_")
        tokens.update(part for part in normalised.split("_") if part)
    return tokens


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
        stripe_width = max(18, round(width * 0.18))
        object_box = {
            "x": stripe_width + round(width * 0.04),
            "y": round(height * 0.08),
            "width": max(1, width - stripe_width - round(width * 0.08)),
            "height": round(height * 0.84),
        }
        stamp_box = {"x": 0, "y": 0, "width": stripe_width, "height": height}
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
        "dataset": "ksdd2",
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
