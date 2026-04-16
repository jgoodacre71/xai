"""Prepared Waterbirds manifest loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WaterbirdsManifestRecord:
    """One prepared Waterbirds sample."""

    dataset: str
    category: str
    split: str
    label: str
    habitat: str
    group: str
    is_aligned: bool
    image_path: Path

    @property
    def sample_id(self) -> str:
        """Return a stable sample id for reports and caching."""

        return f"{self.dataset}/{self.category}/{self.split}/{self.image_path.stem}"


def _resolve_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_root = manifest_path.parent
    while project_root.name != "data" and project_root.parent != project_root:
        project_root = project_root.parent
    if project_root.name == "data":
        project_root = project_root.parent
    return project_root / path


def load_waterbirds_manifest(manifest_path: Path) -> list[WaterbirdsManifestRecord]:
    """Load a canonical Waterbirds JSONL manifest."""

    records: list[WaterbirdsManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                image_path_value = str(raw["image_path"])
                records.append(
                    WaterbirdsManifestRecord(
                        dataset=str(raw["dataset"]),
                        category=str(raw["category"]),
                        split=str(raw["split"]),
                        label=str(raw["label"]),
                        habitat=str(raw["habitat"]),
                        group=str(raw["group"]),
                        is_aligned=bool(raw["is_aligned"]),
                        image_path=_resolve_path(image_path_value, manifest_path),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Manifest {manifest_path} line {line_number} is missing {exc!s}"
                ) from exc
    return records


def filter_waterbirds_records(
    records: list[WaterbirdsManifestRecord],
    *,
    split: str | None = None,
    label: str | None = None,
    habitat: str | None = None,
    aligned: bool | None = None,
) -> list[WaterbirdsManifestRecord]:
    """Filter Waterbirds records by one or more metadata fields."""

    output: list[WaterbirdsManifestRecord] = []
    for record in records:
        if split is not None and record.split != split:
            continue
        if label is not None and record.label != label:
            continue
        if habitat is not None and record.habitat != habitat:
            continue
        if aligned is not None and record.is_aligned != aligned:
            continue
        output.append(record)
    return output
