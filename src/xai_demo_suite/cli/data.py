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
from xai_demo_suite.data.downloaders.mvtec_loco_ad import (
    MVTecLOCOADCategory,
    build_mvtec_loco_ad_manifest,
    download_mvtec_loco_ad_category,
    extract_mvtec_loco_ad_category,
    get_mvtec_loco_ad_category,
    iter_mvtec_loco_ad_categories,
    plan_mvtec_loco_ad_fetch,
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
    fetch_parser.add_argument("dataset", choices=["mvtec_ad", "mvtec_loco_ad"])
    fetch_parser.add_argument("--category", required=True, help="Dataset category name.")
    fetch_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument("--dry-run", action="store_true")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Extract a raw archive and build a processed manifest.",
    )
    prepare_parser.add_argument("dataset", choices=["mvtec_ad", "mvtec_loco_ad"])
    prepare_parser.add_argument("--category", required=True, help="Dataset category name.")
    prepare_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    prepare_parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    prepare_parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    prepare_parser.add_argument("--overwrite", action="store_true")

    return parser


def _print_category(category: MVTecADCategory) -> None:
    print(f"{category.name:12} {category.size_mb:>4} MB  {category.archive_name}")


def _print_loco_category(category: MVTecLOCOADCategory) -> None:
    print(f"{category.name:20} {category.size_mb:>4} MB  {category.archive_name}")


def _handle_list() -> int:
    print("MVTec AD")
    print("  source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
    print("  downloads: https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads")
    print("  licence: CC BY-NC-SA 4.0; non-commercial use only")
    print("  categories:")
    for ad_category in iter_mvtec_ad_categories():
        _print_category(ad_category)
    print()
    print("MVTec LOCO AD")
    print("  source: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad")
    print("  downloads: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad/downloads")
    print("  licence: CC BY-NC-SA 4.0; non-commercial use only")
    print("  categories:")
    for loco_category in iter_mvtec_loco_ad_categories():
        _print_loco_category(loco_category)
    return 0


def _handle_fetch(args: argparse.Namespace) -> int:
    if args.dataset == "mvtec_loco_ad":
        loco_category = get_mvtec_loco_ad_category(args.category)
        loco_plan = plan_mvtec_loco_ad_fetch(
            category=loco_category,
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        if args.dry_run:
            action = "download" if loco_plan.should_download else "skip"
            print(f"{action}: {loco_plan.url}")
            print(f"target: {loco_plan.archive_path}")
            if loco_plan.reason:
                print(f"reason: {loco_plan.reason}")
            return 0
        loco_result = download_mvtec_loco_ad_category(
            category=loco_category,
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        print(f"{loco_result.status}: {loco_result.archive_path}")
        return 0

    ad_category = get_mvtec_ad_category(args.category)
    ad_plan = plan_mvtec_ad_fetch(
        category=ad_category,
        raw_root=args.raw_root,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        action = "download" if ad_plan.should_download else "skip"
        print(f"{action}: {ad_plan.url}")
        print(f"target: {ad_plan.archive_path}")
        if ad_plan.reason:
            print(f"reason: {ad_plan.reason}")
        return 0

    ad_result = download_mvtec_ad_category(
        category=ad_category,
        raw_root=args.raw_root,
        overwrite=args.overwrite,
    )
    print(f"{ad_result.status}: {ad_result.archive_path}")
    return 0


def _handle_prepare(args: argparse.Namespace) -> int:
    if args.dataset == "mvtec_loco_ad":
        loco_category = get_mvtec_loco_ad_category(args.category)
        extracted_root = extract_mvtec_loco_ad_category(
            category=loco_category,
            raw_root=args.raw_root,
            interim_root=args.interim_root,
            overwrite=args.overwrite,
        )
        manifest_path = (
            args.processed_root / "mvtec_loco_ad" / loco_category.name / "manifest.jsonl"
        )
        record_count = build_mvtec_loco_ad_manifest(
            category=loco_category,
            category_root=extracted_root / loco_category.name,
            manifest_path=manifest_path,
            project_root=Path.cwd(),
        )
        print(f"extracted: {extracted_root}")
        print(f"manifest: {manifest_path} ({record_count} records)")
        return 0

    ad_category = get_mvtec_ad_category(args.category)
    extracted_root = extract_mvtec_ad_category(
        category=ad_category,
        raw_root=args.raw_root,
        interim_root=args.interim_root,
        overwrite=args.overwrite,
    )
    manifest_path = args.processed_root / "mvtec_ad" / ad_category.name / "manifest.jsonl"
    record_count = build_mvtec_ad_manifest(
        category=ad_category,
        category_root=extracted_root / ad_category.name,
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
