from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
SVG_DIR = ROOT / "assets" / "equations" / "svg"
PNG_DIR = ROOT / "assets" / "equations" / "png"
MANIFEST = ROOT / "deck" / "asset_manifest.yaml"
REPORT = ROOT / "deck" / "build_report.md"


EQUATIONS = {
    "erm_functional": r"\widehat{R}_n(f)=\frac{1}{n}\sum_{i=1}^{n}\ell(f(x_i),y_i)",
    "training_objective": r"\widehat{f}\in\arg\min_{f\in\mathcal{F}}\widehat{R}_n(f)",
    "many_functions_same_exam": r"\widehat{R}_n(f_{\mathrm{shape}})\approx\widehat{R}_n(f_{\mathrm{position}})\approx\widehat{R}_n(f_{\mathrm{background}})\approx 0",
    "factorised_generator": r"X=R(Y,A,N,\varepsilon)",
    "star_score": r"s_\theta(x)=P_\theta(Y=\mathrm{star}\mid x)",
    "position_counterfactual": r"x(p)=R(y,a,p,\varepsilon),\quad p_{\mathrm{lower\!-\!left}}\longrightarrow p_{\mathrm{upper\!-\!right}}",
    "nuisance_orbit": r"\mathcal{O}_{N}(y,a)=\{R(y,a,n,\varepsilon):n\in\mathcal{N}\}",
    "response_map": r"F_\theta(c_x,c_y)=s_\theta\!\left(R(a,(c_x,c_y),\varepsilon)\right)",
    "decision_contour": r"F_\theta(c_x,c_y)=0.5",
    "position_sensitivity": r"\mathrm{PSI}(F_\theta)=\max_{c_x,c_y}F_\theta(c_x,c_y)-\min_{c_x,c_y}F_\theta(c_x,c_y)",
    "consistency": r"\mathrm{Consistency}=\frac{1}{|\mathcal{G}|}\sum_{(c_x,c_y)\in\mathcal{G}}\mathbf{1}[\widehat{y}_\theta(c_x,c_y)=y]",
    "smoothgrad": r"G_\theta(x)=\frac{1}{K}\sum_{k=1}^{K}\left|\nabla_x s_\theta(x+\eta_k)\right|,\quad \eta_k\sim\mathcal{N}(0,\sigma^2 I)",
    "background_snr": r"\mathrm{SNR}_{\mathrm{mean}}=\frac{\delta\sqrt{n}}{\sigma}",
    "intervention_loop": r"\mathrm{diagnose}\rightarrow\mathrm{intervene}\rightarrow\mathrm{re\!-\!test}",
    "orbit_averaged_risk": r"\min_f\frac{1}{n}\sum_{i=1}^{n}\mathbb{E}_{n\sim\mathcal{N}}[\ell(f(R(y_i,a_i,n)),y_i)]",
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


def add_asset(manifest: dict, asset_id: str, path: Path, asset_type: str) -> None:
    manifest.setdefault("assets", []).append(
        {
            "asset_id": asset_id,
            "filename": path.relative_to(ROOT).as_posix(),
            "type": asset_type,
            "source": "scripts/02_render_equations.py",
            "suggested_use": "Equation asset",
        }
    )


def render_equation(name: str, equation: str) -> tuple[Path, Path]:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = SVG_DIR / f"{name}.svg"
    png_path = PNG_DIR / f"{name}.png"

    width = 15 if len(equation) > 95 else 11
    fig = plt.figure(figsize=(width, 1.55), dpi=220)
    fig.patch.set_alpha(0)
    axis = fig.add_axes([0, 0, 1, 1])
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        f"${equation}$",
        ha="center",
        va="center",
        fontsize=28,
        color="#111111",
    )
    fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, transparent=True, bbox_inches="tight", pad_inches=0.08, dpi=220)
    plt.close(fig)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    return svg_path, png_path


def main() -> None:
    plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    manifest = load_manifest()
    manifest["assets"] = [item for item in manifest.get("assets", []) if not item["asset_id"].startswith("equation_")]
    rendered = []
    for name, equation in EQUATIONS.items():
        svg_path, png_path = render_equation(name, equation)
        add_asset(manifest, f"equation_{name}_svg", svg_path, "svg")
        add_asset(manifest, f"equation_{name}_png", png_path, "png")
        rendered.append(name)
    save_manifest(manifest)
    append_report("# Equation rendering\n\n" + "\n".join(f"- `{name}`" for name in rendered))


if __name__ == "__main__":
    main()
