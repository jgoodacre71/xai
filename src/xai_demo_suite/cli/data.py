"""Command-line interface for local dataset workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_demo_suite.data.downloaders.mvtec_ad import (
    MVTecADCategory,
    build_mvtec_ad_manifest,
    download_mvtec_ad_category,
    extract_mvtec_ad_category,
    get_mvtec_ad_category,
    iter_mvtec_ad_categories,
    plan_mvtec_ad_fetch,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset CLI parser."""

    parser = argparse.ArgumentParser(
        prog="xai-demo-data",
        description="Fetch and prepare local datasets without committing raw data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List supported datasets and categories.")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch a raw dataset archive.")
    fetch_parser.add_argument("dataset", choices=["mvtec_ad"])
    fetch_parser.add_argument("--category", required=True, help="MVTec AD category name.")
    fetch_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument("--dry-run", action="store_true")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Extract a raw archive and build a processed manifest.",
    )
    prepare_parser.add_argument("dataset", choices=["mvtec_ad"])
    prepare_parser.add_argument("--category", required=True, help="MVTec AD category name.")
    prepare_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    prepare_parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    prepare_parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    prepare_parser.add_argument("--overwrite", action="store_true")

    return parser


def _print_category(category: MVTecADCategory) -> None:
    print(f"{category.name:12} {category.size_mb:>4} MB  {category.archive_name}")


def _handle_list() -> int:
    print("MVTec AD")
    print("  source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
    print("  downloads: https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads")
    print("  licence: CC BY-NC-SA 4.0; non-commercial use only")
    print("  categories:")
    for category in iter_mvtec_ad_categories():
        _print_category(category)
    return 0


def _handle_fetch(args: argparse.Namespace) -> int:
    category = get_mvtec_ad_category(args.category)
    plan = plan_mvtec_ad_fetch(
        category=category,
        raw_root=args.raw_root,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        action = "download" if plan.should_download else "skip"
        print(f"{action}: {plan.url}")
        print(f"target: {plan.archive_path}")
        if plan.reason:
            print(f"reason: {plan.reason}")
        return 0

    result = download_mvtec_ad_category(
        category=category,
        raw_root=args.raw_root,
        overwrite=args.overwrite,
    )
    print(f"{result.status}: {result.archive_path}")
    return 0


def _handle_prepare(args: argparse.Namespace) -> int:
    category = get_mvtec_ad_category(args.category)
    extracted_root = extract_mvtec_ad_category(
        category=category,
        raw_root=args.raw_root,
        interim_root=args.interim_root,
        overwrite=args.overwrite,
    )
    manifest_path = args.processed_root / "mvtec_ad" / category.name / "manifest.jsonl"
    record_count = build_mvtec_ad_manifest(
        category=category,
        category_root=extracted_root / category.name,
        manifest_path=manifest_path,
        project_root=Path.cwd(),
    )
    print(f"extracted: {extracted_root}")
    print(f"manifest: {manifest_path} ({record_count} records)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the dataset CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _handle_list()
    if args.command == "fetch":
        return _handle_fetch(args)
    if args.command == "prepare":
        return _handle_prepare(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
