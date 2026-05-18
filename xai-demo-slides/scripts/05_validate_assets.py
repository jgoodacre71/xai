from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SLIDE_MANIFEST = ROOT / "deck" / "slide_manifest.yaml"
ASSET_MANIFEST = ROOT / "deck" / "asset_manifest.yaml"
REPORT = ROOT / "deck" / "build_report.md"
EQUATION_NAMES = [
    "erm_functional",
    "training_objective",
    "many_functions_same_exam",
    "factorised_generator",
    "star_score",
    "position_counterfactual",
    "nuisance_orbit",
    "response_map",
    "decision_contour",
    "position_sensitivity",
    "consistency",
    "smoothgrad",
    "background_snr",
    "intervention_loop",
    "orbit_averaged_risk",
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_report(lines: list[str]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def is_ignored_zip(path: Path) -> bool:
    ignored_parts = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "site-packages"}
    return any(part in ignored_parts for part in path.parts)


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    report: list[str] = ["# Build Report", ""]

    zips = [p for p in REPO_ROOT.rglob("*.zip") if not is_ignored_zip(p)]
    if zips:
        failures.append("Zip files are not allowed: " + ", ".join(p.as_posix() for p in zips))

    for required in [
        ROOT / "deck" / "slide_storyboard.md",
        ROOT / "deck" / "copilot_prompt.md",
        ROOT / "deck" / "design_system.md",
        SLIDE_MANIFEST,
        ASSET_MANIFEST,
    ]:
        if not required.exists():
            failures.append(f"Missing required file: {rel(required)}")

    slides = []
    if SLIDE_MANIFEST.exists():
        slide_data = yaml.safe_load(SLIDE_MANIFEST.read_text(encoding="utf-8")) or {}
        slides = slide_data.get("slides", [])
        for slide in slides:
            for field in ("number", "title", "claim", "speaker_note"):
                if not slide.get(field):
                    failures.append(f"Slide missing `{field}`: {slide}")
            visual_paths = slide.get("image_assets", []) + slide.get("equation_assets", []) + slide.get("video_assets", [])
            if not visual_paths:
                failures.append(f"Slide {slide.get('number')} has no visual or equation asset")
            for key in ("image_assets", "equation_assets", "video_assets", "fallback_assets"):
                for asset in slide.get(key, []):
                    if not (ROOT / asset).exists():
                        failures.append(f"Missing slide asset: {asset}")

    for name in EQUATION_NAMES:
        for suffix in ("svg", "png"):
            path = ROOT / "assets" / "equations" / suffix / f"{name}.{suffix}"
            if not path.exists():
                failures.append(f"Missing equation asset: {rel(path)}")

    gif_dir = ROOT / "assets" / "video" / "gif"
    mp4_dir = ROOT / "assets" / "video" / "mp4"
    first_dir = ROOT / "assets" / "video" / "first_frames"
    for gif in sorted(gif_dir.glob("*.gif")):
        if not (mp4_dir / f"{gif.stem}.mp4").exists():
            failures.append(f"Missing MP4 equivalent for GIF: {gif.name}")
        if not (first_dir / f"{gif.stem}_first_frame.png").exists():
            failures.append(f"Missing first-frame fallback for GIF: {gif.name}")

    for mp4 in sorted(mp4_dir.glob("*.mp4")):
        if not (first_dir / f"{mp4.stem}_first_frame.png").exists():
            failures.append(f"Missing first-frame fallback for MP4: {mp4.name}")

    plate_dir = ROOT / "assets" / "images" / "slide_plates"
    plates = sorted(plate_dir.glob("*.png"))
    if len(plates) < 12:
        failures.append(f"Expected at least 12 slide plates, found {len(plates)}")
    for plate in plates:
        with Image.open(plate) as im:
            if im.size != (1920, 1080):
                failures.append(f"Slide plate is not 1920x1080: {plate.name} is {im.size}")

    media_files = [p for p in ROOT.rglob("*") if p.is_file()]
    large = [p for p in media_files if p.stat().st_size > 50 * 1024 * 1024]
    if large:
        warnings.append("Files over 50 MB: " + ", ".join(rel(p) for p in large))

    name_counts = Counter(p.name for p in media_files)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        warnings.append("Duplicate basenames present: " + ", ".join(sorted(duplicates)))

    report.extend(
        [
            "## Validation summary",
            "",
            f"- slides checked: {len(slides)}",
            f"- slide plates: {len(plates)}",
            f"- GIFs: {len(list(gif_dir.glob('*.gif')))}",
            f"- MP4s: {len(list(mp4_dir.glob('*.mp4')))}",
            f"- equation SVGs: {len(list((ROOT / 'assets' / 'equations' / 'svg').glob('*.svg')))}",
            f"- equation PNGs: {len(list((ROOT / 'assets' / 'equations' / 'png').glob('*.png')))}",
            "",
        ]
    )
    if warnings:
        report.extend(["## Warnings", "", *[f"- {warning}" for warning in warnings], ""])
    if failures:
        report.extend(["## Failures", "", *[f"- {failure}" for failure in failures], ""])
        write_report(report)
        raise SystemExit(1)
    report.extend(["## Result", "", "Validation passed."])
    write_report(report)
    print("Validation passed")


if __name__ == "__main__":
    main()
