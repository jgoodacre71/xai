from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import yaml


ROOT = Path(__file__).resolve().parents[1]
POLISHED = ROOT / "assets" / "images" / "notebook_polished"
EQUATIONS = ROOT / "assets" / "equations" / "png"
PLATES = ROOT / "assets" / "images" / "slide_plates"
MANIFEST = ROOT / "deck" / "asset_manifest.yaml"
REPORT = ROOT / "deck" / "build_report.md"

W, H = 1920, 1080
OFF_WHITE = "#FAFAF8"
CHARCOAL = "#111111"
MUTED = "#555555"
RED = "#B23A48"
GREEN = "#2E8B57"
GOLD = "#D89C21"


PLATE_SPECS = [
    ("plate_01_supervised_problem.png", "Training minimises empirical risk", "Start with the ordinary supervised-learning exam.", "01_train_and_iid_validation_examples.png", "erm_functional.png"),
    ("plate_02_erm_many_functions.png", "Many functions pass the same finite exam", "IID accuracy alone does not identify the human concept.", "02_both_models_pass_iid_evaluation_clean.png", "many_functions_same_exam.png"),
    ("plate_03_iid_success.png", "Both models appear to work", "The trap starts with apparently perfect evaluation.", "02_both_models_pass_iid_evaluation_clean.png", None),
    ("plate_04_position_counterfactual.png", "Same object. Different position. Different answer.", "Changing address changes the MLP score.", "04_counterfactual_probe_static.png", "position_counterfactual.png"),
    ("plate_05_response_map_geometry.png", "Response maps reveal learned geometry", "The score changes over factor space.", "08_response_map_geometry_professional.png", "response_map.png"),
    ("plate_06_position_data_audit.png", "The position shortcut was already in the data", "A rule that ignores shape can pass the biased exam.", "24_data_story_act1_position_shortcut_professional.png", None),
    ("plate_07_cnn_second_comfort.png", "The CNN appears to work again", "A stronger-looking model can still pass the wrong exam.", "12_act2_cnn_appears_to_work_clean.png", None),
    ("plate_08_background_counterfactual.png", "Same shape. Invisible background shift. Different belief.", "The object is fixed; the background statistic changes.", "14_act2_background_swap_reveal_clean.png", "background_snr.png"),
    ("plate_09_xai_heatmap_caution.png", "Heatmaps are supporting evidence, not the explanation", "Counterfactuals identify the controlling factor.", "13_act2_xai_caution_counterfactual_clean.png", "smoothgrad.png"),
    ("plate_10_background_data_audit.png", "Background statistics already encoded the answer", "Human invisibility is not statistical irrelevance.", "25_data_story_act2_background_shortcut_professional.png", "background_snr.png"),
    ("plate_11_mitigation_retest.png", "Intervention means process change plus re-test", "The same behavioural probe checks whether the learned rule changed.", "21_act2_mitigation_retest_clean.png", "intervention_loop.png"),
    ("plate_12_final_synthesis.png", "XAI as experimental model science", "Make learned structure visible, testable, and correctable.", "26_final_demo_synthesis_professional.png", "intervention_loop.png"),
]


def append_report(text: str) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Build Report\n\n"
    REPORT.write_text(existing.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


TITLE = get_font(54, True)
SUBTITLE = get_font(27)
FOOT = get_font(23)


def fit_image(path: Path, box: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    image.thumbnail(box, Image.Resampling.LANCZOS)
    return image


def paste_center(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - image.width) // 2
    y = y0 + (y1 - y0 - image.height) // 2
    canvas.alpha_composite(image, (x, y))


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], outline: str = "#DDDDDD") -> None:
    draw.rounded_rectangle(box, radius=22, fill="white", outline=outline, width=2)


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


def build_plate(filename: str, title: str, subtitle: str, image_name: str, equation_name: str | None) -> Path:
    source_image = POLISHED / image_name
    if not source_image.exists():
        raise FileNotFoundError(source_image)
    equation = EQUATIONS / equation_name if equation_name else None
    if equation is not None and not equation.exists():
        raise FileNotFoundError(equation)

    canvas = Image.new("RGBA", (W, H), OFF_WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 64), title, font=TITLE, fill=CHARCOAL)
    draw.text((94, 137), subtitle, font=SUBTITLE, fill=MUTED)

    accent = RED if "shortcut" in title.lower() or "different" in title.lower() else GREEN if "mitigation" in title.lower() else GOLD if "background" in title.lower() else "#D8D8D8"
    draw.rounded_rectangle((90, 178, 1830, 188), radius=5, fill=accent)

    image_box = (110, 225, 1810, 865) if equation is None else (110, 225, 1810, 780)
    rounded_panel(draw, image_box)
    image = fit_image(source_image, (image_box[2] - image_box[0] - 60, image_box[3] - image_box[1] - 50))
    paste_center(canvas, image, image_box)

    if equation is not None:
        eq_box = (210, 815, 1710, 930)
        rounded_panel(draw, eq_box, "#E7E0CF")
        eq = fit_image(equation, (eq_box[2] - eq_box[0] - 60, eq_box[3] - eq_box[1] - 30))
        paste_center(canvas, eq, eq_box)

    draw.text((110, 995), "Use as a PowerPoint proof object; detailed explanation belongs in speaker notes.", font=FOOT, fill=MUTED)
    path = PLATES / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path, quality=95)
    return path


def main() -> None:
    PLATES.mkdir(parents=True, exist_ok=True)
    for child in PLATES.iterdir():
        if child.is_file():
            child.unlink()
    manifest = load_manifest()
    manifest["assets"] = [item for item in manifest.get("assets", []) if not item["asset_id"].startswith("slide_plate_")]
    created = []
    for spec in PLATE_SPECS:
        path = build_plate(*spec)
        created.append(path.name)
        manifest.setdefault("assets", []).append(
            {
                "asset_id": f"slide_plate_{path.stem}",
                "filename": path.relative_to(ROOT).as_posix(),
                "type": "png",
                "source": "scripts/04_build_slide_plates.py",
                "suggested_use": "PowerPoint slide plate",
            }
        )
    save_manifest(manifest)
    append_report("# Slide plates\n\n" + "\n".join(f"- `{name}`" for name in created))


if __name__ == "__main__":
    main()
