"""Command-line interface for runnable demo/report slices."""

from __future__ import annotations

import argparse
from pathlib import Path

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
from xai_demo_suite.reports.suite import build_demo_suite, verify_demo_suite_outputs
from xai_demo_suite.reports.waterbirds_shortcut import (
    WaterbirdsShortcutReportConfig,
    build_waterbirds_shortcut_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the demo CLI parser."""

    bottle_defaults = PatchCoreBottleReportConfig()
    drift_defaults = ExplanationDriftReportConfig()
    logic_defaults = PatchCoreLogicReportConfig()
    limits_defaults = PatchCoreLimitsReportConfig()
    severity_defaults = PatchCoreSeverityReportConfig()
    wrong_normal_defaults = PatchCoreWrongNormalReportConfig()
    shortcut_defaults = IndustrialShortcutReportConfig()
    waterbirds_defaults = WaterbirdsShortcutReportConfig()
    parser = argparse.ArgumentParser(
        prog="xai-demo-report",
        description="Generate local demo reports from package code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    waterbirds = subparsers.add_parser(
        "waterbirds-shortcut",
        help="Generate the synthetic Waterbirds-style shortcut report.",
    )
    waterbirds.add_argument("--output-dir", type=Path, default=waterbirds_defaults.output_dir)
    waterbirds.add_argument("--synthetic-dir", type=Path, default=waterbirds_defaults.synthetic_dir)

    bottle = subparsers.add_parser(
        "patchcore-bottle",
        help="Generate the first MVTec AD bottle PatchCore report slice.",
    )
    bottle.add_argument("--manifest-path", type=Path, default=bottle_defaults.manifest_path)
    bottle.add_argument("--output-dir", type=Path, default=bottle_defaults.output_dir)
    bottle.add_argument("--cache-path", type=Path, default=bottle_defaults.cache_path)
    bottle.add_argument(
        "--feature-extractor",
        choices=(
            "colour_texture",
            "mean_rgb",
            "resnet18_random",
            "feature_map_resnet18_random",
            "feature_map_resnet18_pretrained",
        ),
        default=bottle_defaults.feature_extractor_name,
        help="Patch feature extractor to use for report generation.",
    )
    bottle.add_argument("--max-train", type=int, default=bottle_defaults.max_train)
    bottle.add_argument("--test-index", type=int, default=bottle_defaults.test_index)
    bottle.add_argument("--max-examples", type=int, default=bottle_defaults.max_examples)
    bottle.add_argument("--patch-size", type=int, default=bottle_defaults.patch_size)
    bottle.add_argument("--stride", type=int, default=bottle_defaults.stride)
    bottle.add_argument("--top-k", type=int, default=bottle_defaults.top_k)
    bottle.add_argument("--input-size", type=int, default=bottle_defaults.input_size)
    bottle.add_argument("--batch-size", type=int, default=bottle_defaults.batch_size)
    bottle.add_argument("--coreset-size", type=int, default=bottle_defaults.coreset_size)
    bottle.add_argument("--coreset-seed", type=int, default=bottle_defaults.coreset_seed)
    bottle.add_argument(
        "--benchmark-limit",
        type=int,
        default=bottle_defaults.max_benchmark_records,
        help="Optional cap on MVTec bottle test records scored for report diagnostics.",
    )
    bottle.add_argument("--no-cache", action="store_true")

    limits = subparsers.add_parser(
        "patchcore-limits",
        help="Generate the synthetic PatchCore limits report.",
    )
    limits.add_argument("--output-dir", type=Path, default=limits_defaults.output_dir)
    limits.add_argument("--cache-path", type=Path, default=limits_defaults.cache_path)
    limits.add_argument("--synthetic-dir", type=Path, default=limits_defaults.synthetic_dir)
    limits.add_argument("--patch-size", type=int, default=limits_defaults.patch_size)
    limits.add_argument("--stride", type=int, default=limits_defaults.stride)
    limits.add_argument("--top-k", type=int, default=limits_defaults.top_k)
    limits.add_argument("--no-cache", action="store_true")

    severity = subparsers.add_parser(
        "patchcore-severity",
        help="Generate the synthetic PatchCore severity report.",
    )
    severity.add_argument("--output-dir", type=Path, default=severity_defaults.output_dir)
    severity.add_argument("--cache-path", type=Path, default=severity_defaults.cache_path)
    severity.add_argument("--synthetic-dir", type=Path, default=severity_defaults.synthetic_dir)
    severity.add_argument("--patch-size", type=int, default=severity_defaults.patch_size)
    severity.add_argument("--stride", type=int, default=severity_defaults.stride)
    severity.add_argument("--top-k", type=int, default=severity_defaults.top_k)
    severity.add_argument("--no-cache", action="store_true")

    logic = subparsers.add_parser(
        "patchcore-logic",
        help="Generate the synthetic PatchCore logical anomaly report.",
    )
    logic.add_argument("--output-dir", type=Path, default=logic_defaults.output_dir)
    logic.add_argument("--cache-path", type=Path, default=logic_defaults.cache_path)
    logic.add_argument("--synthetic-dir", type=Path, default=logic_defaults.synthetic_dir)
    logic.add_argument("--patch-size", type=int, default=logic_defaults.patch_size)
    logic.add_argument("--stride", type=int, default=logic_defaults.stride)
    logic.add_argument("--top-k", type=int, default=logic_defaults.top_k)
    logic.add_argument("--no-cache", action="store_true")

    wrong_normal = subparsers.add_parser(
        "patchcore-wrong-normal",
        help="Generate the synthetic PatchCore wrong-normal report.",
    )
    wrong_normal.add_argument("--output-dir", type=Path, default=wrong_normal_defaults.output_dir)
    wrong_normal.add_argument(
        "--synthetic-dir",
        type=Path,
        default=wrong_normal_defaults.synthetic_dir,
    )
    wrong_normal.add_argument(
        "--clean-cache-path",
        type=Path,
        default=wrong_normal_defaults.clean_cache_path,
    )
    wrong_normal.add_argument(
        "--contaminated-cache-path",
        type=Path,
        default=wrong_normal_defaults.contaminated_cache_path,
    )
    wrong_normal.add_argument("--patch-size", type=int, default=wrong_normal_defaults.patch_size)
    wrong_normal.add_argument("--stride", type=int, default=wrong_normal_defaults.stride)
    wrong_normal.add_argument("--top-k", type=int, default=wrong_normal_defaults.top_k)
    wrong_normal.add_argument("--no-cache", action="store_true")

    shortcut = subparsers.add_parser(
        "shortcut-industrial",
        help="Generate the synthetic industrial shortcut report.",
    )
    shortcut.add_argument("--output-dir", type=Path, default=shortcut_defaults.output_dir)
    shortcut.add_argument("--synthetic-dir", type=Path, default=shortcut_defaults.synthetic_dir)

    drift = subparsers.add_parser(
        "explanation-drift",
        help="Generate the synthetic explanation drift report.",
    )
    drift.add_argument("--output-dir", type=Path, default=drift_defaults.output_dir)
    drift.add_argument("--synthetic-dir", type=Path, default=drift_defaults.synthetic_dir)

    suite = subparsers.add_parser(
        "suite",
        help="Generate the current demo suite.",
    )
    suite.add_argument("--output-root", type=Path, default=Path("outputs"))
    suite.add_argument("--include-mvtec", action="store_true")
    suite.add_argument("--mvtec-manifest-path", type=Path, default=None)
    suite.add_argument(
        "--mvtec-feature-extractor",
        choices=(
            "colour_texture",
            "mean_rgb",
            "resnet18_random",
            "feature_map_resnet18_random",
            "feature_map_resnet18_pretrained",
        ),
        default=None,
        help=(
            "Feature extractor for the optional MVTec bottle report. "
            "Pretrained weights are only used when feature_map_resnet18_pretrained "
            "is selected."
        ),
    )
    suite.add_argument("--mvtec-max-train", type=int, default=None)
    suite.add_argument("--mvtec-max-examples", type=int, default=None)
    suite.add_argument("--mvtec-coreset-size", type=int, default=None)
    suite.add_argument("--mvtec-input-size", type=int, default=None)
    suite.add_argument("--mvtec-benchmark-limit", type=int, default=None)
    suite.add_argument("--no-cache", action="store_true")

    verify = subparsers.add_parser(
        "verify",
        help="Verify generated reports, demo cards, figures, and local index.",
    )
    verify.add_argument("--output-root", type=Path, default=Path("outputs"))

    return parser


def _handle_patchcore_bottle(args: argparse.Namespace) -> int:
    config = PatchCoreBottleReportConfig(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        feature_extractor_name=args.feature_extractor,
        max_train=args.max_train,
        test_index=args.test_index,
        max_examples=args.max_examples,
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        input_size=args.input_size,
        batch_size=args.batch_size,
        coreset_size=args.coreset_size,
        coreset_seed=args.coreset_seed,
        max_benchmark_records=args.benchmark_limit,
        use_cache=not args.no_cache,
    )
    output_path = build_patchcore_bottle_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_waterbirds_shortcut(args: argparse.Namespace) -> int:
    config = WaterbirdsShortcutReportConfig(
        output_dir=args.output_dir,
        synthetic_dir=args.synthetic_dir,
    )
    output_path = build_waterbirds_shortcut_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_patchcore_limits(args: argparse.Namespace) -> int:
    config = PatchCoreLimitsReportConfig(
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        synthetic_dir=args.synthetic_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    output_path = build_patchcore_limits_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_patchcore_severity(args: argparse.Namespace) -> int:
    config = PatchCoreSeverityReportConfig(
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        synthetic_dir=args.synthetic_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    output_path = build_patchcore_severity_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_patchcore_logic(args: argparse.Namespace) -> int:
    config = PatchCoreLogicReportConfig(
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        synthetic_dir=args.synthetic_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    output_path = build_patchcore_logic_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_patchcore_wrong_normal(args: argparse.Namespace) -> int:
    config = PatchCoreWrongNormalReportConfig(
        output_dir=args.output_dir,
        synthetic_dir=args.synthetic_dir,
        clean_cache_path=args.clean_cache_path,
        contaminated_cache_path=args.contaminated_cache_path,
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    output_path = build_patchcore_wrong_normal_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_shortcut_industrial(args: argparse.Namespace) -> int:
    config = IndustrialShortcutReportConfig(
        output_dir=args.output_dir,
        synthetic_dir=args.synthetic_dir,
    )
    output_path = build_industrial_shortcut_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_explanation_drift(args: argparse.Namespace) -> int:
    config = ExplanationDriftReportConfig(
        output_dir=args.output_dir,
        synthetic_dir=args.synthetic_dir,
    )
    output_path = build_explanation_drift_report(config)
    print(f"report: {output_path}")
    return 0


def _handle_suite(args: argparse.Namespace) -> int:
    results = build_demo_suite(
        output_root=args.output_root,
        include_mvtec=args.include_mvtec,
        use_cache=not args.no_cache,
        mvtec_manifest_path=args.mvtec_manifest_path,
        mvtec_feature_extractor_name=args.mvtec_feature_extractor,
        mvtec_max_train=args.mvtec_max_train,
        mvtec_max_examples=args.mvtec_max_examples,
        mvtec_coreset_size=args.mvtec_coreset_size,
        mvtec_input_size=args.mvtec_input_size,
        mvtec_benchmark_limit=args.mvtec_benchmark_limit,
    )
    failed = False
    for result in results:
        if result.output_path is None:
            failed = True
            print(f"{result.status}: {result.name}: {result.note}")
        else:
            print(f"{result.status}: {result.name}: {result.output_path}")
    return 1 if failed else 0


def _handle_verify(args: argparse.Namespace) -> int:
    result = verify_demo_suite_outputs(args.output_root)
    print(f"cards: {result.card_count}")
    print(f"checked paths: {len(result.checked_paths)}")
    if result.ok:
        print("verification: ok")
        return 0
    print("verification: failed")
    for problem in result.problems:
        print(f"- {problem}")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the demo/report CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "waterbirds-shortcut":
        return _handle_waterbirds_shortcut(args)
    if args.command == "patchcore-bottle":
        return _handle_patchcore_bottle(args)
    if args.command == "patchcore-limits":
        return _handle_patchcore_limits(args)
    if args.command == "patchcore-severity":
        return _handle_patchcore_severity(args)
    if args.command == "patchcore-logic":
        return _handle_patchcore_logic(args)
    if args.command == "patchcore-wrong-normal":
        return _handle_patchcore_wrong_normal(args)
    if args.command == "shortcut-industrial":
        return _handle_shortcut_industrial(args)
    if args.command == "explanation-drift":
        return _handle_explanation_drift(args)
    if args.command == "suite":
        return _handle_suite(args)
    if args.command == "verify":
        return _handle_verify(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
