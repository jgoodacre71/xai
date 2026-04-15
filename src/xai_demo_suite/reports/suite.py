"""Suite-level report generation and verification helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from xai_demo_suite.reports.cards import save_demo_index_for_output_root
from xai_demo_suite.reports.explanation_drift import (
    ExplanationDriftReportConfig,
    build_explanation_drift_report,
)
from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    build_patchcore_bottle_report,
)
from xai_demo_suite.reports.patchcore_limits import (
    PatchCoreLimitsReportConfig,
    build_patchcore_limits_report,
)
from xai_demo_suite.reports.patchcore_wrong_normal import (
    PatchCoreWrongNormalReportConfig,
    build_patchcore_wrong_normal_report,
)
from xai_demo_suite.reports.shortcut_industrial import (
    IndustrialShortcutReportConfig,
    build_industrial_shortcut_report,
)
from xai_demo_suite.reports.waterbirds_shortcut import (
    WaterbirdsShortcutReportConfig,
    build_waterbirds_shortcut_report,
)


@dataclass(frozen=True, slots=True)
class SuiteBuildResult:
    """Result for one attempted report build."""

    name: str
    output_path: Path | None
    status: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class SuiteVerificationResult:
    """Verification summary for generated demo suite outputs."""

    output_root: Path
    card_count: int
    checked_paths: tuple[Path, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether the generated suite passed verification."""

        return not self.problems


ReportBuilder = Callable[[], Path]


def build_demo_suite(
    *,
    output_root: Path = Path("outputs"),
    include_mvtec: bool = False,
    use_cache: bool = True,
) -> tuple[SuiteBuildResult, ...]:
    """Build the current local demo suite.

    Synthetic reports are always generated. The MVTec bottle report is opt-in
    because fresh clones will not have the external dataset prepared.
    """

    builders: list[tuple[str, ReportBuilder]] = [
        (
            "waterbirds-shortcut",
            lambda: build_waterbirds_shortcut_report(
                WaterbirdsShortcutReportConfig(
                    output_dir=output_root / "waterbirds_shortcut",
                    synthetic_dir=output_root / "waterbirds_shortcut" / "synthetic",
                )
            ),
        ),
        (
            "shortcut-industrial",
            lambda: build_industrial_shortcut_report(
                IndustrialShortcutReportConfig(
                    output_dir=output_root / "shortcut_industrial",
                    synthetic_dir=output_root / "shortcut_industrial" / "synthetic",
                )
            ),
        ),
        (
            "patchcore-limits",
            lambda: build_patchcore_limits_report(
                PatchCoreLimitsReportConfig(
                    output_dir=output_root / "patchcore_limits",
                    synthetic_dir=output_root / "patchcore_limits" / "synthetic",
                    use_cache=use_cache,
                )
            ),
        ),
        (
            "patchcore-wrong-normal",
            lambda: build_patchcore_wrong_normal_report(
                PatchCoreWrongNormalReportConfig(
                    output_dir=output_root / "patchcore_wrong_normal",
                    synthetic_dir=output_root / "patchcore_wrong_normal" / "synthetic",
                    use_cache=use_cache,
                )
            ),
        ),
        (
            "explanation-drift",
            lambda: build_explanation_drift_report(
                ExplanationDriftReportConfig(
                    output_dir=output_root / "explanation_drift",
                    synthetic_dir=output_root / "explanation_drift" / "synthetic",
                )
            ),
        ),
    ]
    if include_mvtec:
        builders.append(
            (
                "patchcore-bottle",
                lambda: build_patchcore_bottle_report(
                    PatchCoreBottleReportConfig(
                        output_dir=output_root / "patchcore_bottle",
                        use_cache=use_cache,
                    )
                ),
            )
        )

    results: list[SuiteBuildResult] = []
    for name, builder in builders:
        try:
            results.append(SuiteBuildResult(name=name, output_path=builder(), status="built"))
        except Exception as exc:  # pragma: no cover - exercised through CLI failure path.
            results.append(
                SuiteBuildResult(
                    name=name,
                    output_path=None,
                    status="failed",
                    note=str(exc),
                )
            )
    save_demo_index_for_output_root(output_root)
    return tuple(results)


def verify_demo_suite_outputs(output_root: Path = Path("outputs")) -> SuiteVerificationResult:
    """Verify generated reports, demo cards, figures, and local index."""

    problems: list[str] = []
    checked_paths: list[Path] = []
    index_path = output_root / "index.html"
    _record_path(index_path, checked_paths, problems, "local index")
    card_paths = sorted(output_root.glob("*/demo_card.json"))
    if not card_paths:
        problems.append(f"No demo cards found under {output_root}.")

    for card_path in card_paths:
        _record_path(card_path, checked_paths, problems, "demo card JSON")
        html_card = card_path.with_suffix(".html")
        _record_path(html_card, checked_paths, problems, "demo card HTML")
        if not card_path.exists():
            continue
        data = _load_card_json(card_path, problems)
        if data is None:
            continue
        report_path = output_root / str(data.get("report_path", ""))
        _record_path(report_path, checked_paths, problems, f"report for {card_path.parent.name}")
        for figure in data.get("figure_paths", []):
            _record_path(
                output_root / str(figure),
                checked_paths,
                problems,
                f"figure for {card_path.parent.name}",
            )

    return SuiteVerificationResult(
        output_root=output_root,
        card_count=len(card_paths),
        checked_paths=tuple(checked_paths),
        problems=tuple(problems),
    )


def _load_card_json(card_path: Path, problems: list[str]) -> dict[str, Any] | None:
    try:
        raw_data = json.loads(card_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"Invalid JSON in {card_path}: {exc}")
        return None
    if not isinstance(raw_data, dict):
        problems.append(f"{card_path} must contain a JSON object.")
        return None
    data = cast(dict[str, Any], raw_data)
    required = {
        "title",
        "task",
        "model",
        "explanation_methods",
        "key_lesson",
        "failure_mode",
        "intervention",
        "remaining_caveats",
        "report_path",
        "figure_paths",
    }
    missing = sorted(required - set(data))
    if missing:
        problems.append(f"{card_path} is missing fields: {', '.join(missing)}")
    return data


def _record_path(
    path: Path,
    checked_paths: list[Path],
    problems: list[str],
    description: str,
) -> None:
    checked_paths.append(path)
    if not path.exists():
        problems.append(f"Missing {description}: {path}")
