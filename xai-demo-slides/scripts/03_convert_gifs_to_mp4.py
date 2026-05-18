from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageSequence
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
GIF_DIR = ROOT / "assets" / "video" / "gif"
MP4_DIR = ROOT / "assets" / "video" / "mp4"
FIRST_FRAME_DIR = ROOT / "assets" / "video" / "first_frames"
MANIFEST = ROOT / "deck" / "asset_manifest.yaml"
REPORT = ROOT / "deck" / "build_report.md"

SEARCH_DIRS = [
    GIF_DIR,
    ROOT / "outputs" / "demo00_story_assets",
    REPO_ROOT / "notebooks" / "outputs" / "demo00_story_assets",
]

ALIASES = {
    "anim_01_moon_moves_confidence": "anim_moon_moves_confidence",
    "anim_02_star_moves_confidence": "anim_star_moves_confidence",
    "anim_11_invisible_background_morph_moon": "anim_invisible_background_morph_moon",
    "anim_12_invisible_background_morph_star": "anim_invisible_background_morph_star",
    "anim_13_shape_morph_moon_to_star": "anim_shape_morph_moon_to_star",
    "anim_14_shape_morph_star_to_moon": "anim_shape_morph_star_to_moon",
    "anim_15_moon_movement_heatmaps": "anim_moon_movement_heatmaps",
    "anim_16_star_movement_heatmaps": "anim_star_movement_heatmaps",
}


def append_report(text: str) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Build Report\n\n"
    REPORT.write_text(existing.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def load_manifest() -> dict:
    if MANIFEST.exists():
        return yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {"assets": []}
    return {"assets": []}


def save_manifest(data: dict) -> None:
    unique = {}
    for item in data.get("assets", []):
        unique[item["asset_id"]] = item
    data["assets"] = list(unique.values())
    MANIFEST.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def add_asset(manifest: dict, asset_id: str, path: Path, asset_type: str, source: str) -> None:
    manifest.setdefault("assets", []).append(
        {
            "asset_id": asset_id,
            "filename": path.relative_to(ROOT).as_posix(),
            "type": asset_type,
            "source": source,
            "suggested_use": "PowerPoint animation asset" if asset_type == "mp4" else "Animation first-frame fallback",
        }
    )


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def collect_gifs() -> dict[str, Path]:
    gifs: dict[str, Path] = {}
    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.gif")):
            gifs.setdefault(path.name, path)
    return gifs


def ensure_even_dimensions(gif_path: Path) -> tuple[int, int]:
    with Image.open(gif_path) as im:
        width, height = im.size
    return width - (width % 2), height - (height % 2)


def convert_gif(gif_path: Path, mp4_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for GIF to MP4 conversion")
    width, height = ensure_even_dimensions(gif_path)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(gif_path),
        "-vf",
        f"fps=10,scale={width}:{height}:flags=lanczos,format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def write_first_frame(gif_path: Path, output_path: Path) -> None:
    with Image.open(gif_path) as im:
        frame = next(ImageSequence.Iterator(im)).convert("RGBA")
        background = Image.new("RGBA", frame.size, "WHITE")
        background.alpha_composite(frame)
        background.convert("RGB").save(output_path)


def main() -> None:
    clean_dir(MP4_DIR)
    clean_dir(FIRST_FRAME_DIR)
    manifest = load_manifest()
    manifest["assets"] = [item for item in manifest.get("assets", []) if not item["asset_id"].startswith(("mp4_", "first_frame_"))]
    converted = []
    gifs = collect_gifs()
    for name, gif_path in gifs.items():
        stem = gif_path.stem
        mp4_path = MP4_DIR / f"{stem}.mp4"
        first_frame = FIRST_FRAME_DIR / f"{stem}_first_frame.png"
        convert_gif(gif_path, mp4_path)
        write_first_frame(gif_path, first_frame)
        add_asset(manifest, f"mp4_{stem}", mp4_path, "mp4", gif_path.as_posix())
        add_asset(manifest, f"first_frame_{stem}", first_frame, "png", gif_path.as_posix())
        converted.append(stem)

        alias = ALIASES.get(stem)
        if alias:
            alias_mp4 = MP4_DIR / f"{alias}.mp4"
            alias_frame = FIRST_FRAME_DIR / f"{alias}_first_frame.png"
            shutil.copy2(mp4_path, alias_mp4)
            shutil.copy2(first_frame, alias_frame)
            add_asset(manifest, f"mp4_{alias}", alias_mp4, "mp4", gif_path.as_posix())
            add_asset(manifest, f"first_frame_{alias}", alias_frame, "png", gif_path.as_posix())
            converted.append(alias)

    save_manifest(manifest)
    append_report("# GIF to MP4 conversion\n\n" + "\n".join(f"- `{name}`" for name in converted))


if __name__ == "__main__":
    main()
