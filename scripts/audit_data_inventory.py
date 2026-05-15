#!/usr/bin/env python3
"""Audit local dataset inventory against the repository registry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathCheck:
    """A local path check derived from one registry entry."""

    label: str
    path: str
    exists: bool | None
    note: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit local dataset paths against data_registry.yaml.",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data_registry.yaml"),
        help="Registry path relative to --root unless absolute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output report path. When omitted, the report is printed.",
    )
    return parser.parse_args()


def load_registry(path: Path) -> dict[str, Any]:
    """Load the YAML dataset registry."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Registry {path} does not contain a 'datasets' mapping.")
    return data


def has_placeholder(path_text: str) -> bool:
    """Return whether a registry path contains an unresolved placeholder."""

    return "<" in path_text or ">" in path_text


def check_path(root: Path, label: str, path_text: str) -> PathCheck:
    """Check one local path unless it contains placeholders."""

    if has_placeholder(path_text):
        return PathCheck(
            label=label,
            path=path_text,
            exists=None,
            note="pattern contains a placeholder; check a concrete prepared category.",
        )
    candidate = root / path_text
    return PathCheck(
        label=label,
        path=path_text,
        exists=candidate.exists(),
        note="found" if candidate.exists() else "missing",
    )


def iter_path_checks(root: Path, dataset: dict[str, Any]) -> list[PathCheck]:
    """Build path checks for the expected layout fields in a registry row."""

    checks: list[PathCheck] = []
    expected_layout = dataset.get("expected_layout", {})
    if not isinstance(expected_layout, dict):
        return checks
    for label, path_value in expected_layout.items():
        if isinstance(path_value, str) and path_value.startswith("data/"):
            checks.append(check_path(root=root, label=label, path_text=path_value))
    return checks


def format_bool(value: bool | None) -> str:
    """Format tri-state path existence for a text report."""

    if value is None:
        return "pattern"
    return "yes" if value else "no"


def build_report(root: Path, registry_path: Path, registry: dict[str, Any]) -> str:
    """Build a plain-text local inventory report."""

    datasets = registry["datasets"]
    lines = [
        "# Local data inventory",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Repository root: {root.resolve()}",
        f"Registry: {registry_path.resolve()}",
        "",
        "| Dataset | Status | Work use status | Path label | Exists | Path / note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for dataset_id, dataset in sorted(datasets.items()):
        if not isinstance(dataset, dict):
            continue
        name = str(dataset.get("name", dataset_id))
        status = str(dataset.get("status", "unknown"))
        work_status = str(
            dataset.get("work_use_status", dataset.get("usage_restriction", "unknown"))
        )
        checks = iter_path_checks(root=root, dataset=dataset)
        if not checks:
            lines.append(
                f"| {name} | {status} | {work_status} | none | n/a | no local path declared |"
            )
            continue
        for check in checks:
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        status,
                        work_status,
                        check.label,
                        format_bool(check.exists),
                        f"`{check.path}` - {check.note}",
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("This report records local file presence only. It is not a licence approval.")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the inventory audit."""

    args = parse_args()
    root = args.root.resolve()
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    registry = load_registry(registry_path)
    report = build_report(root=root, registry_path=registry_path, registry=registry)
    if args.output:
        output_path = args.output if args.output.is_absolute() else root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
