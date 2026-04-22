from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebooks_are_output_free() -> None:
    notebooks = sorted(Path("notebooks").rglob("*.ipynb"))
    assert notebooks

    for notebook_path in notebooks:
        notebook = _load_notebook(notebook_path)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell.get("execution_count") is None, notebook_path
            assert cell.get("outputs") == [], notebook_path


def test_required_demo_notebooks_exist() -> None:
    notebooks = {
        path.relative_to("notebooks").as_posix()
        for path in Path("notebooks").rglob("*.ipynb")
    }

    assert {
        "overview/00_overview.ipynb",
        "shortcut_lab/01_waterbirds_shortcut.ipynb",
        "shortcut_lab/02_industrial_shortcut_trap.ipynb",
        "patchcore_explainability/03_patchcore_mvtec_ad.ipynb",
        "patchcore_explainability/04_patchcore_wrong_normal.ipynb",
        "patchcore_limits/05_patchcore_count_limit.ipynb",
        "patchcore_limits/06_patchcore_severity_limit.ipynb",
        "patchcore_limits/07_patchcore_loco_logic_limit.ipynb",
        "robustness_drift/08_explanation_drift.ipynb",
        "global_local_explainability/09_global_vs_local_explainability_shap.ipynb",
    }.issubset(notebooks)


def test_notebooks_follow_shared_narrative_template() -> None:
    required_sections = (
        "## Learning goals",
        "## Why this demo matters",
        "## Dataset and task definition",
        "## Model and explanation methods",
        "## Baseline result",
        "## Failure or pitfall",
        "## Intervention",
        "## Re-test",
        "## What we learned",
        "## Residual risks and next questions",
    )

    template_notebooks = (
        Path("notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb"),
        Path("notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb"),
        Path("notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb"),
        Path("notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb"),
        Path("notebooks/patchcore_limits/05_patchcore_count_limit.ipynb"),
        Path("notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb"),
        Path("notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb"),
        Path("notebooks/robustness_drift/08_explanation_drift.ipynb"),
    )

    for notebook_path in template_notebooks:
        notebook = _load_notebook(notebook_path)
        text = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for section in required_sections:
            assert section in text, notebook_path


def test_active_demo_notebooks_are_self_contained() -> None:
    active_notebooks = (
        Path("notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb"),
        Path("notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb"),
        Path("notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb"),
        Path("notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb"),
        Path("notebooks/patchcore_limits/05_patchcore_count_limit.ipynb"),
        Path("notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb"),
        Path("notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb"),
        Path("notebooks/robustness_drift/08_explanation_drift.ipynb"),
    )

    forbidden_strings = (
        "xai_demo_suite.reports",
        "xai_demo_suite.notebooks",
        "outputs/",
        "build_waterbirds_shortcut_report",
        "build_industrial_shortcut_report",
        "build_patchcore_bottle_report",
        "build_patchcore_wrong_normal_report",
        "build_patchcore_limits_report",
        "build_patchcore_severity_report",
        "build_patchcore_logic_report",
        "build_explanation_drift_report",
    )

    for notebook_path in active_notebooks:
        notebook = _load_notebook(notebook_path)
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "from xai_demo_suite" not in code, notebook_path
        for forbidden in forbidden_strings:
            assert forbidden not in code, notebook_path


def test_demo01_is_real_data_only_and_has_no_cartoon_shortcut_helpers() -> None:
    notebook_path = Path("notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb")
    notebook = _load_notebook(notebook_path)
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "DATA_MODE = 'real'" in code
    assert "MANIFEST_RELATIVE_PATH = Path('data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl')" in code
    assert "PROJECT_ROOT = find_project_root()" in code
    assert "ResNet18_Weights.DEFAULT" in code
    assert "resnet18(weights=None)" in code
    assert "StandardScaler()" in code
    assert "LogisticRegression(max_iter=2000, random_state=SEED)" in code
    assert "manifest_exists: {MANIFEST_EXISTS}" in code

    forbidden_strings = (
        "def render_synthetic_record",
        "def build_synthetic_records",
        "synthetic_group_specs =",
        "synthetic_seed =",
        "TinyWaterbirdsCNN",
    )
    full_notebook_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] in {"code", "markdown"}
    )
    for forbidden in forbidden_strings:
        assert forbidden not in full_notebook_text, notebook_path
