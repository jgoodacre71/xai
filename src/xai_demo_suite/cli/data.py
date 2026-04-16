"""Command-line interface for local dataset workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_demo_suite.data.downloaders.metashift import (
    MetaShiftDataset,
    build_metashift_manifest,
    get_metashift_dataset,
    iter_metashift_datasets,
    metashift_source_root,
    plan_metashift_fetch,
)
from xai_demo_suite.data.downloaders.mvtec_ad import (
    MVTecADCategory,
    build_mvtec_ad_manifest,
    download_mvtec_ad_category,
    extract_mvtec_ad_category,
    get_mvtec_ad_category,
    iter_mvtec_ad_categories,
    plan_mvtec_ad_fetch,
)
from xai_demo_suite.data.downloaders.mvtec_ad_2 import (
    MVTecAD2Dataset,
    build_mvtec_ad_2_manifests,
    download_mvtec_ad_2_dataset,
    extract_mvtec_ad_2_dataset,
    get_mvtec_ad_2_dataset,
    iter_mvtec_ad_2_datasets,
    mvtec_ad_2_archive_dir,
    plan_mvtec_ad_2_fetch,
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
from xai_demo_suite.data.downloaders.visa import (
    VisADataset,
    build_visa_manifests,
    download_visa_dataset,
    download_visa_split_csv,
    extract_visa_dataset,
    get_visa_dataset,
    iter_visa_datasets,
    plan_visa_fetch,
    prepare_visa_one_class_layout,
    visa_split_csv_path,
)
from xai_demo_suite.data.downloaders.waterbirds import (
    WaterbirdsCategory,
    build_waterbirds_manifest,
    download_waterbirds_category,
    extract_waterbirds_category,
    get_waterbirds_category,
    iter_waterbirds_categories,
    plan_waterbirds_fetch,
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
    fetch_parser.add_argument(
        "dataset",
        choices=["metashift", "mvtec_ad", "mvtec_ad_2", "mvtec_loco_ad", "visa", "waterbirds"],
    )
    fetch_parser.add_argument("--category", required=True, help="Dataset category name.")
    fetch_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    fetch_parser.add_argument("--external-root", type=Path, default=Path("data/external"))
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument("--dry-run", action="store_true")
    fetch_parser.add_argument(
        "--archive-url",
        help="Explicit direct archive URL for datasets whose official page does not expose one.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Extract a raw archive and build a processed manifest.",
    )
    prepare_parser.add_argument(
        "dataset",
        choices=["metashift", "mvtec_ad", "mvtec_ad_2", "mvtec_loco_ad", "visa", "waterbirds"],
    )
    prepare_parser.add_argument("--category", required=True, help="Dataset category name.")
    prepare_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    prepare_parser.add_argument("--external-root", type=Path, default=Path("data/external"))
    prepare_parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    prepare_parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument(
        "--archive-path",
        type=Path,
        help=(
            "Optional explicit archive path when the raw archive directory contains "
            "multiple files."
        ),
    )
    prepare_parser.add_argument(
        "--split-csv-path",
        type=Path,
        help="Optional explicit VisA one-class split CSV path.",
    )
    prepare_parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional explicit source root for manually prepared external datasets.",
    )

    return parser


def _print_category(category: MVTecADCategory) -> None:
    print(f"{category.name:12} {category.size_mb:>4} MB  {category.archive_name}")


def _print_loco_category(category: MVTecLOCOADCategory) -> None:
    print(f"{category.name:20} {category.size_mb:>4} MB  {category.archive_name}")


def _print_waterbirds_category(category: WaterbirdsCategory) -> None:
    print(f"{category.name:32} {category.archive_name}")


def _print_mvtec_ad_2_dataset(dataset: MVTecAD2Dataset) -> None:
    print(f"{dataset.name:12} {dataset.scenario_count:>2} scenarios  discovered at prepare time")


def _print_metashift_dataset(dataset: MetaShiftDataset) -> None:
    print(f"{dataset.name:40} {dataset.expected_root_name}")


def _print_visa_dataset(dataset: VisADataset) -> None:
    print(f"{dataset.name:12} {dataset.subset_count:>2} categories  {dataset.archive_name}")


def _handle_list() -> int:
    print("MVTec AD")
    print("  source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
    print("  downloads: https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads")
    print("  licence: CC BY-NC-SA 4.0; non-commercial use only")
    print("  categories:")
    for ad_category in iter_mvtec_ad_categories():
        _print_category(ad_category)
    print()
    print("MVTec AD 2")
    print("  source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2")
    print("  downloads: official page is gated; use --archive-url or place a local archive")
    print("  licence: CC BY-NC-SA 4.0; non-commercial use only")
    print("  datasets:")
    for ad2_dataset in iter_mvtec_ad_2_datasets():
        _print_mvtec_ad_2_dataset(ad2_dataset)
    print()
    print("MetaShift")
    print("  source: https://github.com/Weixin-Liang/MetaShift")
    print("  docs: https://metashift.readthedocs.io/en/latest/sub_pages/applications.html")
    print("  downloads: generated locally from upstream scripts and Visual Genome / GQA assets")
    print("  licence: MIT repository; verify upstream Visual Genome / GQA terms before use")
    print("  datasets:")
    for metashift_dataset in iter_metashift_datasets():
        _print_metashift_dataset(metashift_dataset)
    print()
    print("VisA")
    print("  source: https://github.com/amazon-science/spot-diff")
    print(
        "  downloads: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
    )
    print("  licence: CC BY 4.0")
    print("  datasets:")
    for visa_dataset in iter_visa_datasets():
        _print_visa_dataset(visa_dataset)
    print()
    print("MVTec LOCO AD")
    print("  source: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad")
    print("  downloads: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad/downloads")
    print("  licence: CC BY-NC-SA 4.0; non-commercial use only")
    print("  categories:")
    for loco_category in iter_mvtec_loco_ad_categories():
        _print_loco_category(loco_category)
    print()
    print("Waterbirds")
    print("  source: https://github.com/kohpangwei/group_DRO")
    print("  downloads: https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz")
    print("  licence: verify upstream CUB and Places terms before use")
    print("  usage: treat as research-only unless upstream terms clearly allow more")
    print("  categories:")
    for waterbirds_category in iter_waterbirds_categories():
        _print_waterbirds_category(waterbirds_category)
    return 0


def _handle_fetch(args: argparse.Namespace) -> int:
    if args.dataset == "metashift":
        metashift_dataset = get_metashift_dataset(args.category)
        plan = plan_metashift_fetch(
            dataset=metashift_dataset,
            external_root=args.external_root,
        )
        print("manual: MetaShift does not provide a single raw archive through this CLI")
        print(f"target root: {plan.target_root}")
        print(f"reason: {plan.reason}")
        return 0

    if args.dataset == "visa":
        visa_dataset = get_visa_dataset(args.category)
        visa_plan = plan_visa_fetch(
            dataset=visa_dataset,
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        split_csv_target = visa_split_csv_path(args.raw_root)
        if args.dry_run:
            action = "download" if visa_plan.should_download else "skip"
            print(f"{action}: {visa_plan.url}")
            print(f"target: {visa_plan.archive_path}")
            print(f"split csv: {split_csv_target}")
            if visa_plan.reason:
                print(f"reason: {visa_plan.reason}")
            return 0
        visa_result = download_visa_dataset(
            dataset=visa_dataset,
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        split_csv_file = download_visa_split_csv(
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        print(f"{visa_result.status}: {visa_result.archive_path}")
        print(f"split csv: {split_csv_file}")
        return 0

    if args.dataset == "mvtec_ad_2":
        ad2_dataset = get_mvtec_ad_2_dataset(args.category)
        ad2_plan = plan_mvtec_ad_2_fetch(
            dataset=ad2_dataset,
            raw_root=args.raw_root,
            archive_url=args.archive_url,
            overwrite=args.overwrite,
        )
        if args.dry_run:
            if ad2_plan.should_download:
                print(f"download: {ad2_plan.url}")
                print(f"target: {ad2_plan.archive_path}")
            else:
                print("manual: supply --archive-url or place a single archive under")
                print(f"target dir: {mvtec_ad_2_archive_dir(args.raw_root)}")
            print(f"reason: {ad2_plan.reason}")
            return 0
        if args.archive_url is None:
            raise ValueError(
                "MVTec AD 2 fetch requires --archive-url, or place the archive manually and run "
                "prepare with --archive-path if needed."
            )
        ad2_result = download_mvtec_ad_2_dataset(
            dataset=ad2_dataset,
            raw_root=args.raw_root,
            archive_url=args.archive_url,
            overwrite=args.overwrite,
        )
        print(f"{ad2_result.status}: {ad2_result.archive_path}")
        return 0

    if args.dataset == "waterbirds":
        waterbirds_category = get_waterbirds_category(args.category)
        waterbirds_plan = plan_waterbirds_fetch(
            category=waterbirds_category,
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        if args.dry_run:
            action = "download" if waterbirds_plan.should_download else "skip"
            print(f"{action}: {waterbirds_plan.url}")
            print(f"target: {waterbirds_plan.archive_path}")
            if waterbirds_plan.reason:
                print(f"reason: {waterbirds_plan.reason}")
            return 0
        waterbirds_result = download_waterbirds_category(
            category=waterbirds_category,
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        print(f"{waterbirds_result.status}: {waterbirds_result.archive_path}")
        return 0

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
    if args.dataset == "metashift":
        metashift_dataset = get_metashift_dataset(args.category)
        source_root = args.source_root or metashift_source_root(
            args.external_root,
            metashift_dataset,
        )
        manifest_path = (
            args.processed_root / "metashift" / metashift_dataset.name / "manifest.jsonl"
        )
        record_count = build_metashift_manifest(
            dataset=metashift_dataset,
            source_root=source_root,
            manifest_path=manifest_path,
            project_root=Path.cwd(),
        )
        print(f"source: {source_root}")
        print(f"manifest: {manifest_path} ({record_count} records)")
        return 0

    if args.dataset == "visa":
        visa_dataset = get_visa_dataset(args.category)
        extracted_root = extract_visa_dataset(
            dataset=visa_dataset,
            raw_root=args.raw_root,
            interim_root=args.interim_root,
            overwrite=args.overwrite,
        )
        split_csv_file = args.split_csv_path or visa_split_csv_path(args.raw_root)
        prepared_root = prepare_visa_one_class_layout(
            extracted_root=extracted_root,
            interim_root=args.interim_root,
            split_csv_path=split_csv_file,
            overwrite=args.overwrite,
        )
        category_counts = build_visa_manifests(
            prepared_root=prepared_root,
            processed_root=args.processed_root,
            project_root=Path.cwd(),
        )
        print(f"extracted: {extracted_root}")
        print(f"prepared: {prepared_root}")
        print(f"category manifests: {len(category_counts)}")
        for category_name, record_count in sorted(category_counts.items()):
            manifest_path = args.processed_root / "visa" / category_name / "manifest.jsonl"
            print(f"manifest: {manifest_path} ({record_count} records)")
        return 0

    if args.dataset == "mvtec_ad_2":
        get_mvtec_ad_2_dataset(args.category)
        extracted_root = extract_mvtec_ad_2_dataset(
            raw_root=args.raw_root,
            interim_root=args.interim_root,
            archive_path=args.archive_path,
            overwrite=args.overwrite,
        )
        scenario_counts = build_mvtec_ad_2_manifests(
            extracted_root=extracted_root,
            processed_root=args.processed_root,
            project_root=Path.cwd(),
        )
        print(f"extracted: {extracted_root}")
        print(f"scenario manifests: {len(scenario_counts)}")
        for scenario_name, record_count in sorted(scenario_counts.items()):
            manifest_path = args.processed_root / "mvtec_ad_2" / scenario_name / "manifest.jsonl"
            print(f"manifest: {manifest_path} ({record_count} records)")
        return 0

    if args.dataset == "waterbirds":
        waterbirds_category = get_waterbirds_category(args.category)
        extracted_root = extract_waterbirds_category(
            category=waterbirds_category,
            raw_root=args.raw_root,
            interim_root=args.interim_root,
            overwrite=args.overwrite,
        )
        manifest_path = (
            args.processed_root / "waterbirds" / waterbirds_category.name / "manifest.jsonl"
        )
        record_count = build_waterbirds_manifest(
            category=waterbirds_category,
            category_root=extracted_root / waterbirds_category.name,
            manifest_path=manifest_path,
            project_root=Path.cwd(),
        )
        print(f"extracted: {extracted_root}")
        print(f"manifest: {manifest_path} ({record_count} records)")
        return 0

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
