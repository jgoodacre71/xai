from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
EXECUTED_NOTEBOOK = ROOT / "assets" / "xai_demo_executed.ipynb"
SOURCE_NOTEBOOK = ROOT / "xai_demo.ipynb"
RAW_DIR = ROOT / "assets" / "images" / "notebook_raw"
POLISHED_DIR = ROOT / "assets" / "images" / "notebook_polished"
GIF_DIR = ROOT / "assets" / "video" / "gif"
MANIFEST = ROOT / "deck" / "asset_manifest.yaml"
REPORT = ROOT / "deck" / "build_report.md"

SOURCE_ASSET_DIRS = [
    ROOT / "outputs" / "demo00_story_assets",
    REPO_ROOT / "notebooks" / "outputs" / "demo00_story_assets",
]


SUGGESTED_USE = {
    "01_train_and_iid_validation_examples.png": "Supervised learning setup",
    "02_both_models_pass_iid_evaluation_clean.png": "IID success before reveal",
    "04_counterfactual_probe_static.png": "Same object, different position",
    "08_response_map_geometry_professional.png": "Response maps",
    "14_act2_background_swap_reveal_clean.png": "Same object, background shift",
    "13_act2_xai_caution_counterfactual_clean.png": "Heatmaps are not enough",
    "21_act2_mitigation_retest_clean.png": "Mitigation and re-test",
    "26_final_demo_synthesis_professional.png": "XAI as experimental model science",
    "24_data_story_act1_position_shortcut_professional.png": "Position data audit",
    "25_data_story_act2_background_shortcut_professional.png": "Background data audit",
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
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    assets = data.get("assets", [])
    unique = {}
    for item in assets:
        unique[item["asset_id"]] = item
    data["assets"] = list(unique.values())
    MANIFEST.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def add_asset(manifest: dict, *, asset_id: str, path: Path, asset_type: str, source: str, cell_index: int | None = None, suggested_use: str = "") -> None:
    item = {
        "asset_id": asset_id,
        "filename": path.as_posix(),
        "type": asset_type,
        "source": source,
        "suggested_use": suggested_use,
    }
    if cell_index is not None:
        item["notebook_cell_index"] = cell_index
    manifest.setdefault("assets", []).append(item)


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def payload_text(value) -> str:
    if isinstance(value, list):
        return "".join(value)
    return str(value)


def safe_stem(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", Path(name).stem).strip("_").lower() or "asset"


def copy_polished_assets(manifest: dict) -> tuple[int, int]:
    png_count = 0
    gif_count = 0
    for source_dir in SOURCE_ASSET_DIRS:
        if not source_dir.exists():
            continue
        for source in sorted(source_dir.iterdir()):
            if not source.is_file():
                continue
            if source.suffix.lower() == ".png":
                target = POLISHED_DIR / source.name
                shutil.copy2(source, target)
                add_asset(
                    manifest,
                    asset_id=f"notebook_polished_{safe_stem(source.name)}",
                    path=target.relative_to(ROOT),
                    asset_type="png",
                    source=source.as_posix(),
                    suggested_use=SUGGESTED_USE.get(source.name, "Notebook polished figure"),
                )
                png_count += 1
            elif source.suffix.lower() == ".gif":
                target = GIF_DIR / source.name
                shutil.copy2(source, target)
                add_asset(
                    manifest,
                    asset_id=f"gif_{safe_stem(source.name)}",
                    path=target.relative_to(ROOT),
                    asset_type="gif",
                    source=source.as_posix(),
                    suggested_use=SUGGESTED_USE.get(source.name, "Notebook animation"),
                )
                gif_count += 1
        if png_count or gif_count:
            break
    return png_count, gif_count


def extract_embedded_outputs(manifest: dict) -> tuple[int, int]:
    notebook = EXECUTED_NOTEBOOK if EXECUTED_NOTEBOOK.exists() else SOURCE_NOTEBOOK
    nb = nbformat.read(notebook, as_version=4)
    image_count = 0
    gif_count = 0
    for cell_index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if not isinstance(data, dict):
                continue
            for mime, suffix in (("image/png", "png"), ("image/jpeg", "jpg")):
                if mime in data:
                    raw = payload_text(data[mime])
                    name = f"cell_{cell_index:03d}_output_{output_index:02d}.{suffix}"
                    target = RAW_DIR / name
                    target.write_bytes(base64.b64decode(raw))
                    add_asset(
                        manifest,
                        asset_id=f"notebook_raw_{safe_stem(name)}",
                        path=target.relative_to(ROOT),
                        asset_type=suffix,
                        source=notebook.as_posix(),
                        cell_index=cell_index,
                        suggested_use="Raw notebook output for review",
                    )
                    image_count += 1
            html = data.get("text/html")
            if html:
                html_text = payload_text(html)
                match = re.search(r"data:image/gif;base64,([^\"']+)", html_text)
                if match:
                    name = f"embedded_cell_{cell_index:03d}_output_{output_index:02d}.gif"
                    target = GIF_DIR / name
                    target.write_bytes(base64.b64decode(match.group(1)))
                    add_asset(
                        manifest,
                        asset_id=f"embedded_gif_{safe_stem(name)}",
                        path=target.relative_to(ROOT),
                        asset_type="gif",
                        source=notebook.as_posix(),
                        cell_index=cell_index,
                        suggested_use="Embedded notebook animation",
                    )
                    gif_count += 1
    return image_count, gif_count


def main() -> None:
    for path in (RAW_DIR, POLISHED_DIR, GIF_DIR):
        clean_dir(path)
    manifest = load_manifest()
    manifest["assets"] = [item for item in manifest.get("assets", []) if not item["asset_id"].startswith(("notebook_", "gif_", "embedded_gif_"))]
    polished_pngs, polished_gifs = copy_polished_assets(manifest)
    raw_images, embedded_gifs = extract_embedded_outputs(manifest)
    save_manifest(manifest)
    append_report(
        "# Notebook output extraction\n\n"
        f"- polished PNGs copied: {polished_pngs}\n"
        f"- polished GIFs copied: {polished_gifs}\n"
        f"- embedded raw images extracted: {raw_images}\n"
        f"- embedded GIFs extracted: {embedded_gifs}\n"
        f"- manifest: `{MANIFEST}`"
    )


if __name__ == "__main__":
    main()
