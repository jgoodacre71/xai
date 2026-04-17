"""Manifest helpers for real-image industrial shortcut demos."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from xai_demo_suite.data.synthetic import IndustrialShortcutSample
from xai_demo_suite.explain.contracts import BoundingBox


@dataclass(frozen=True, slots=True)
class IndustrialShortcutManifestRecord:
    """One prepared industrial shortcut sample backed by a real source image."""

    dataset: str
    category: str
    split: str
    label: str
    original_class: str
    variant: str
    stamp: str
    is_aligned: bool
    image_path: Path
    object_region: BoundingBox
    stamp_region: BoundingBox

    @property
    def sample_id(self) -> str:
        """Return a stable sample identifier."""

        return (
            f"{self.dataset}/{self.category}/{self.split}/"
            f"{self.label}/{self.image_path.stem}"
        )


def load_industrial_shortcut_manifest(
    manifest_path: Path,
) -> list[IndustrialShortcutManifestRecord]:
    """Load a JSONL manifest written by the industrial shortcut preparer."""

    records: list[IndustrialShortcutManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                image_path = _resolve_manifest_path(str(raw["image_path"]), manifest_path)
                object_box = _parse_box(raw["object_region"])
                stamp_box = _parse_box(raw["stamp_region"])
                if image_path is None:
                    raise ValueError("image_path must resolve to a file path.")
                records.append(
                    IndustrialShortcutManifestRecord(
                        dataset=str(raw["dataset"]),
                        category=str(raw["category"]),
                        split=str(raw["split"]),
                        label=str(raw["label"]),
                        original_class=str(raw["original_class"]),
                        variant=str(raw["variant"]),
                        stamp=str(raw["stamp"]),
                        is_aligned=bool(raw["is_aligned"]),
                        image_path=image_path,
                        object_region=object_box,
                        stamp_region=stamp_box,
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Manifest {manifest_path} line {line_number} is missing {exc!s}"
                ) from exc
    return records


def manifest_records_to_samples(
    records: list[IndustrialShortcutManifestRecord],
) -> list[IndustrialShortcutSample]:
    """Adapt real-image manifest records to the shared industrial sample shape."""

    return [
        IndustrialShortcutSample(
            sample_id=record.sample_id,
            image_path=record.image_path,
            split=record.split,
            label=record.label,
            shape=record.original_class,
            stamp=record.stamp,
            object_region=record.object_region,
            stamp_region=record.stamp_region,
            variant=record.variant,
        )
        for record in records
    ]


def balanced_label_subset(
    samples: list[IndustrialShortcutSample],
    max_records: int | None,
) -> list[IndustrialShortcutSample]:
    """Return a roughly balanced label subset while preserving local order."""

    if max_records is None or max_records >= len(samples):
        return list(samples)
    if max_records <= 0:
        return []

    grouped: dict[str, list[IndustrialShortcutSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    labels = sorted(grouped)
    if not labels:
        return []

    selected: list[IndustrialShortcutSample] = []
    quota = max_records // len(labels)
    remainder = max_records % len(labels)
    for index, label in enumerate(labels):
        limit = quota + (1 if index < remainder else 0)
        selected.extend(grouped[label][:limit])

    if len(selected) < max_records:
        selected_ids = {sample.sample_id for sample in selected}
        for sample in samples:
            if sample.sample_id in selected_ids:
                continue
            selected.append(sample)
            if len(selected) >= max_records:
                break

    selected_set = {sample.sample_id for sample in selected[:max_records]}
    return [sample for sample in samples if sample.sample_id in selected_set][:max_records]


def _parse_box(raw_box: dict[str, int]) -> BoundingBox:
    return BoundingBox(
        x=int(raw_box["x"]),
        y=int(raw_box["y"]),
        width=int(raw_box["width"]),
        height=int(raw_box["height"]),
    )


def _resolve_manifest_path(path_value: str | None, manifest_path: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_root = manifest_path.parent
    while project_root.name != "data" and project_root.parent != project_root:
        project_root = project_root.parent
    if project_root.name == "data":
        project_root = project_root.parent
    return project_root / path
