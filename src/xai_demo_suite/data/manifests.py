"""Dataset manifest loading helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImageManifestRecord:
    """One image entry from a prepared dataset manifest."""

    dataset: str
    category: str
    split: str
    defect_type: str
    is_anomalous: bool
    image_path: Path
    mask_path: Path | None

    @property
    def sample_id(self) -> str:
        """Return a stable sample id derived from dataset metadata and path."""

        return (
            f"{self.dataset}/{self.category}/{self.split}/"
            f"{self.defect_type}/{self.image_path.stem}"
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


def load_image_manifest(manifest_path: Path) -> list[ImageManifestRecord]:
    """Load a JSONL image manifest written by dataset preparation code."""

    records: list[ImageManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                records.append(
                    ImageManifestRecord(
                        dataset=str(raw["dataset"]),
                        category=str(raw["category"]),
                        split=str(raw["split"]),
                        defect_type=str(raw["defect_type"]),
                        is_anomalous=bool(raw["is_anomalous"]),
                        image_path=_resolve_manifest_path(
                            str(raw["image_path"]),
                            manifest_path,
                        )
                        or Path(),
                        mask_path=_resolve_manifest_path(raw.get("mask_path"), manifest_path),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Manifest {manifest_path} line {line_number} is missing {exc!s}"
                ) from exc
    return records


def filter_manifest_records(
    records: Iterable[ImageManifestRecord],
    *,
    split: str | None = None,
    defect_type: str | None = None,
    is_anomalous: bool | None = None,
) -> list[ImageManifestRecord]:
    """Return records matching optional manifest filters."""

    output: list[ImageManifestRecord] = []
    for record in records:
        if split is not None and record.split != split:
            continue
        if defect_type is not None and record.defect_type != defect_type:
            continue
        if is_anomalous is not None and record.is_anomalous != is_anomalous:
            continue
        output.append(record)
    return output
