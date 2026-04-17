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
from xai_demo_suite.reports.patchcore_logic import (
    PatchCoreLogicReportConfig,
    build_patchcore_logic_report,
)
from xai_demo_suite.reports.patchcore_severity import (
    PatchCoreSeverityReportConfig,
    build_patchcore_severity_report,
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

REQUIRED_CARD_FIELDS: frozenset[str] = frozenset(
    {
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
)

REPORT_MARKERS: dict[str, tuple[str, ...]] = {
    "waterbirds_shortcut": ("Waterbirds", "Shortcut", "<title>"),
    "shortcut_industrial": ("industrial", "shortcut", "<title>"),
    "patchcore_bottle": ("PatchCore", "<title>"),
    "patchcore_limits": ("PatchCore", "count", "<title>"),
    "patchcore_severity": ("PatchCore", "severity", "<title>"),
    "patchcore_logic": ("PatchCore", "logic", "<title>"),
    "patchcore_wrong_normal": ("wrong normal", "PatchCore", "<title>"),
    "explanation_drift": ("drift", "explanation", "<title>"),
}


def _use_local_data_defaults(output_root: Path) -> bool:
    """Return whether suite defaults should auto-pick prepared local datasets."""

    return output_root.resolve() == Path("outputs").resolve()


def _disabled_manifest_path(output_root: Path, name: str) -> Path:
    """Return a guaranteed-missing manifest path for synthetic-only suite builds."""

    return output_root / "_disabled" / name / "manifest.jsonl"


def build_demo_suite(
    *,
    output_root: Path = Path("outputs"),
    include_mvtec: bool = False,
    use_cache: bool = True,
    mvtec_manifest_path: Path | None = None,
    mvtec_feature_extractor_name: str | None = None,
    mvtec_max_train: int | None = None,
    mvtec_max_examples: int | None = None,
    mvtec_coreset_size: int | None = None,
    mvtec_input_size: int | None = None,
    mvtec_benchmark_limit: int | None = None,
    visa_processed_root: Path | None = None,
) -> tuple[SuiteBuildResult, ...]:
    """Build the current local demo suite.

    Synthetic reports are always generated. The MVTec bottle report is opt-in
    because fresh clones will not have the external dataset prepared.
    """

    use_local_data_defaults = _use_local_data_defaults(output_root)

    builders: list[tuple[str, ReportBuilder]] = [
        (
            "waterbirds-shortcut",
            lambda: build_waterbirds_shortcut_report(
                WaterbirdsShortcutReportConfig(
                    output_dir=output_root / "waterbirds_shortcut",
                    synthetic_dir=output_root / "waterbirds_shortcut" / "synthetic",
                    use_real_data=use_local_data_defaults,
                )
            ),
        ),
        (
            "shortcut-industrial",
            lambda: build_industrial_shortcut_report(
                IndustrialShortcutReportConfig(
                    output_dir=output_root / "shortcut_industrial",
                    synthetic_dir=output_root / "shortcut_industrial" / "synthetic",
                    use_real_data=use_local_data_defaults,
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
            "patchcore-severity",
            lambda: build_patchcore_severity_report(
                PatchCoreSeverityReportConfig(
                    output_dir=output_root / "patchcore_severity",
                    synthetic_dir=output_root / "patchcore_severity" / "synthetic",
                    use_cache=use_cache,
                )
            ),
        ),
        (
            "patchcore-logic",
            lambda: build_patchcore_logic_report(
                PatchCoreLogicReportConfig(
                    output_dir=output_root / "patchcore_logic",
                    synthetic_dir=output_root / "patchcore_logic" / "synthetic",
                    manifest_path=(
                        PatchCoreLogicReportConfig().manifest_path
                        if use_local_data_defaults
                        else _disabled_manifest_path(output_root, "patchcore_logic")
                    ),
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
                    industrial_manifest_path=(
                        ExplanationDriftReportConfig().industrial_manifest_path
                        if use_local_data_defaults
                        else _disabled_manifest_path(output_root, "industrial_shortcut")
                    ),
                    include_mvtec_if_available=use_local_data_defaults,
                    visa_processed_root=(
                        visa_processed_root
                        if visa_processed_root is not None
                        else ExplanationDriftReportConfig().visa_processed_root
                    ),
                )
            ),
        ),
    ]
    if include_mvtec:
        bottle_defaults = PatchCoreBottleReportConfig()
        builders.append(
            (
                "patchcore-bottle",
                lambda: build_patchcore_bottle_report(
                    PatchCoreBottleReportConfig(
                        manifest_path=mvtec_manifest_path or bottle_defaults.manifest_path,
                        output_dir=output_root / "patchcore_bottle",
                        feature_extractor_name=(
                            mvtec_feature_extractor_name
                            or bottle_defaults.feature_extractor_name
                        ),
                        max_train=mvtec_max_train or bottle_defaults.max_train,
                        max_examples=mvtec_max_examples or bottle_defaults.max_examples,
                        input_size=mvtec_input_size or bottle_defaults.input_size,
                        coreset_size=mvtec_coreset_size,
                        max_benchmark_records=mvtec_benchmark_limit,
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
    """Verify generated reports, demo cards, figures, and basic report semantics."""

    problems: list[str] = []
    checked_paths: list[Path] = []
    index_path = output_root / "index.html"
    _record_path(index_path, checked_paths, problems, "local index")
    if index_path.exists():
        _check_html_contains(
            index_path,
            ("XAI Demo Suite Local Reports", "Open report"),
            problems,
            "local index",
        )
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
        _check_card_fields(card_path, data, problems)
        if html_card.exists():
            _check_html_contains(
                html_card,
                (str(data.get("title", "")), "Key lesson", "Open report"),
                problems,
                f"demo card HTML for {card_path.parent.name}",
            )
        report_path = output_root / str(data.get("report_path", ""))
        _record_path(report_path, checked_paths, problems, f"report for {card_path.parent.name}")
        if report_path.exists():
            _check_report_semantics(report_path, problems)
        for figure in data.get("figure_paths", []):
            _record_path(
                output_root / str(figure),
                checked_paths,
                problems,
                f"figure for {card_path.parent.name}",
            )

    review_pack_path = output_root / "review_pack" / "index.html"
    if review_pack_path.exists():
        _record_path(review_pack_path, checked_paths, problems, "review pack")
        _check_html_contains(
            review_pack_path,
            ("XAI Demo Suite Review Pack", "ChatGPT Handoff", "Recommended Walkthrough"),
            problems,
            "review pack",
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
    missing = sorted(REQUIRED_CARD_FIELDS - set(data))
    if missing:
        problems.append(f"{card_path} is missing fields: {', '.join(missing)}")
    return data


def _check_card_fields(card_path: Path, data: dict[str, Any], problems: list[str]) -> None:
    string_fields = ("title", "task", "model", "key_lesson", "failure_mode", "intervention")
    for field in string_fields:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{card_path} field '{field}' must be a non-empty string.")

    list_fields = ("explanation_methods", "remaining_caveats", "figure_paths")
    for field in list_fields:
        value = data.get(field)
        if not isinstance(value, list) or not value:
            problems.append(f"{card_path} field '{field}' must be a non-empty list.")


def _check_report_semantics(report_path: Path, problems: list[str]) -> None:
    slug = report_path.parent.name
    markers = REPORT_MARKERS.get(slug, ("<title>",))
    _check_html_contains(report_path, markers, problems, f"report {slug}")


def _check_html_contains(
    path: Path,
    markers: tuple[str, ...],
    problems: list[str],
    description: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    lowered_text = text.lower()
    for marker in markers:
        if marker.lower() not in lowered_text:
            problems.append(f"Missing marker '{marker}' in {description}: {path}")


def _record_path(
    path: Path,
    checked_paths: list[Path],
    problems: list[str],
    description: str,
) -> None:
    checked_paths.append(path)
    if not path.exists():
        problems.append(f"Missing {description}: {path}")
