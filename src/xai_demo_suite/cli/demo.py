"""Command-line interface for runnable demo/report slices."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    """Build the demo CLI parser."""

    bottle_defaults = PatchCoreBottleReportConfig()
    limits_defaults = PatchCoreLimitsReportConfig()
    wrong_normal_defaults = PatchCoreWrongNormalReportConfig()
    shortcut_defaults = IndustrialShortcutReportConfig()
    parser = argparse.ArgumentParser(
        prog="xai-demo-report",
        description="Generate local demo reports from package code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bottle = subparsers.add_parser(
        "patchcore-bottle",
        help="Generate the first MVTec AD bottle PatchCore report slice.",
    )
    bottle.add_argument("--manifest-path", type=Path, default=bottle_defaults.manifest_path)
    bottle.add_argument("--output-dir", type=Path, default=bottle_defaults.output_dir)
    bottle.add_argument("--cache-path", type=Path, default=bottle_defaults.cache_path)
    bottle.add_argument(
        "--feature-extractor",
        choices=("colour_texture", "mean_rgb", "resnet18_random"),
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
        use_cache=not args.no_cache,
    )
    output_path = build_patchcore_bottle_report(config)
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


def main(argv: list[str] | None = None) -> int:
    """Run the demo/report CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "patchcore-bottle":
        return _handle_patchcore_bottle(args)
    if args.command == "patchcore-limits":
        return _handle_patchcore_limits(args)
    if args.command == "patchcore-wrong-normal":
        return _handle_patchcore_wrong_normal(args)
    if args.command == "shortcut-industrial":
        return _handle_shortcut_industrial(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
