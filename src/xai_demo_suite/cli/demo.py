"""Command-line interface for runnable demo/report slices."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    build_patchcore_bottle_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the demo CLI parser."""

    defaults = PatchCoreBottleReportConfig()
    parser = argparse.ArgumentParser(
        prog="xai-demo-report",
        description="Generate local demo reports from package code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bottle = subparsers.add_parser(
        "patchcore-bottle",
        help="Generate the first MVTec AD bottle PatchCore report slice.",
    )
    bottle.add_argument("--manifest-path", type=Path, default=defaults.manifest_path)
    bottle.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    bottle.add_argument("--cache-path", type=Path, default=defaults.cache_path)
    bottle.add_argument("--max-train", type=int, default=defaults.max_train)
    bottle.add_argument("--test-index", type=int, default=defaults.test_index)
    bottle.add_argument("--patch-size", type=int, default=defaults.patch_size)
    bottle.add_argument("--stride", type=int, default=defaults.stride)
    bottle.add_argument("--top-k", type=int, default=defaults.top_k)
    bottle.add_argument("--input-size", type=int, default=defaults.input_size)
    bottle.add_argument("--batch-size", type=int, default=defaults.batch_size)
    bottle.add_argument("--no-cache", action="store_true")

    return parser


def _handle_patchcore_bottle(args: argparse.Namespace) -> int:
    config = PatchCoreBottleReportConfig(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        max_train=args.max_train,
        test_index=args.test_index,
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


def main(argv: list[str] | None = None) -> int:
    """Run the demo/report CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "patchcore-bottle":
        return _handle_patchcore_bottle(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
