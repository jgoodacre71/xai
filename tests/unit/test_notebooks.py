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
        "shortcut_lab/00_moons_stars_clever_hans.ipynb",
        "shortcut_lab/01_waterbirds_shortcut.ipynb",
        "shortcut_lab/02_industrial_shortcut_trap.ipynb",
        "patchcore_explainability/03_patchcore_mvtec_ad.ipynb",
        "patchcore_explainability/04_patchcore_wrong_normal.ipynb",
        "patchcore_limits/05_patchcore_count_limit.ipynb",
        "patchcore_limits/06_patchcore_severity_limit.ipynb",
        "patchcore_limits/07_patchcore_loco_logic_limit.ipynb",
        "robustness_drift/08_explanation_drift.ipynb",
        "data_scouting/90_ieee_dataset_scouting.ipynb",
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
        Path("notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb"),
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
        Path("notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb"),
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
        "outputs/reports",
        "outputs/html",
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


def test_shortcut_and_scouting_notebooks_show_data_status() -> None:
    required_status_fields = (
        "DEMO",
        "DATA_MODE",
        "EXTERNAL_DATA_REQUIRED",
        "MANIFEST_PATH",
        "MANIFEST_EXISTS",
        "PROJECT_ROOT",
        "DATASET_SOURCE",
        "LICENCE_NOTE",
        "MISSING_FILES",
        "SEED",
    )

    status_notebooks = (
        Path("notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb"),
        Path("notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb"),
        Path("notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb"),
        Path("notebooks/patchcore_explainability/03_patchcore_mvtec_ad.ipynb"),
        Path("notebooks/patchcore_explainability/04_patchcore_wrong_normal.ipynb"),
        Path("notebooks/patchcore_limits/05_patchcore_count_limit.ipynb"),
        Path("notebooks/patchcore_limits/06_patchcore_severity_limit.ipynb"),
        Path("notebooks/patchcore_limits/07_patchcore_loco_logic_limit.ipynb"),
        Path("notebooks/robustness_drift/08_explanation_drift.ipynb"),
        Path("notebooks/data_scouting/90_ieee_dataset_scouting.ipynb"),
    )

    for notebook_path in status_notebooks:
        notebook = _load_notebook(notebook_path)
        text = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] in {"code", "markdown"}
        )
        for field in required_status_fields:
            assert field in text, notebook_path


def test_demo00_is_generated_controlled_demo_with_no_external_data() -> None:
    notebook_path = Path("notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb")
    notebook = _load_notebook(notebook_path)
    text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] in {"code", "markdown"}
    )

    assert 'DATA_MODE = "generated_controlled_demo"' in text
    assert "EXTERNAL_DATA_REQUIRED = False" in text
    assert "Generated inside this notebook" in text
    assert "PixelMLP" in text
    assert "GapCNN" in text
    assert "position-augmented train" in text
    assert "absolute position" in text
    assert "fig08_shape_morph_strip.png" in text
    assert "fig09_shape_position_surface.png" in text
    assert "fig10_movement_path.png" in text
    assert "fig11_saliency_comparison.png" in text
    assert "fig12_average_relevance_maps.png" in text
    assert "fig13_representation_neighbours.png" in text
    assert "fig14_representation_probes.png" in text
    assert "fig15_minimal_evidence_removal.png" in text
    assert "fig16_what_changes_the_decision.png" in text
    assert "fig17_evidence_ledger.png" in text
    assert "movement_path_results" in text
    assert "shape_morph_results" in text
    assert "shape_position_surface_mlp" in text
    assert "position_response_metrics" in text
    assert "decision_boundary_results" in text
    assert "attribution_density_results" in text
    assert "average_relevance_results" in text
    assert "representation_neighbour_results" in text
    assert "representation_probe_results" in text
    assert "mlp_cf_predictions == [0, 1, 1, 0]" in text
    assert "cnn_cf_predictions == [0, 0, 1, 1]" in text
    assert "mlp_crossed_accuracy <= 0.35" in text
    assert "cnn_crossed_accuracy >= 0.90" in text
    assert "Demo 00 self-check passed." in text
    assert "from xai_demo_suite" not in text

    forbidden_old_shortcut_phrases = (
        "blue scene",
        "amber scene",
        "moon on blue",
        "moon on amber",
        "star on blue",
        "star on amber",
        "SCENE_NAMES",
        "scene cue",
    )
    for phrase in forbidden_old_shortcut_phrases:
        assert phrase not in text, phrase


def test_demo01_is_real_data_only_and_has_no_cartoon_shortcut_helpers() -> None:
    notebook_path = Path("notebooks/shortcut_lab/01_waterbirds_shortcut.ipynb")
    notebook = _load_notebook(notebook_path)
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "DATA_MODE = 'real'" in code
    assert (
        "MANIFEST_RELATIVE_PATH = "
        "Path('data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl')"
    ) in code
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


def test_demo02_is_real_data_only_and_has_no_toy_shortcut_helpers() -> None:
    notebook_path = Path("notebooks/shortcut_lab/02_industrial_shortcut_trap.ipynb")
    notebook = _load_notebook(notebook_path)
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "DATA_MODE = 'real_neu_controlled_shortcut'" in code
    assert (
        "MANIFEST_RELATIVE_PATH = "
        "Path('data/processed/neu_cls/shortcut_binary/manifest.jsonl')"
    ) in code
    assert "PROJECT_ROOT = find_project_root()" in code
    assert "ResNet18_Weights.DEFAULT" in code
    assert "resnet18(weights=None)" in code
    assert "StandardScaler()" in code
    assert "LogisticRegression(max_iter=2000, random_state=SEED)" in code
    assert "manifest_exists: {MANIFEST_EXISTS}" in code

    forbidden_strings = (
        "def render_panel",
        "def make_samples",
        "stamp_shortcut_score",
        "shape_score",
        "PartSample",
    )
    full_notebook_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] in {"code", "markdown"}
    )
    for forbidden in forbidden_strings:
        assert forbidden not in full_notebook_text, notebook_path
