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

    demo00 = _load_notebook(Path("notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb"))
    demo00_text = "\n".join(
        "".join(cell["source"])
        for cell in demo00["cells"]
        if cell["cell_type"] == "markdown"
    )
    for section in (
        "## 1. Title and hook",
        "## 2. The apparent task",
        "## 4. Both models appear to work",
        "## 5. The first crack: same object, different position",
        "## 7. The hidden exam leak",
        "## 8. Response maps: the model has learned geography",
        "## 7. Act II: The CNN learns a different shortcut",
        "## 11. Final evidence board and bridge",
        "## 12. What we learned",
        "## Residual risks and next questions",
    ):
        assert section in demo00_text


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
    assert "act1_margin_loss" in text
    assert "ACT1_CONFIDENCE_MEAN_THRESHOLD = 0.98" in text
    assert "ACT1_CONFIDENCE_MIN_THRESHOLD = 0.90" in text
    assert "cnn_iid_validation_mean_correct_prob >= 0.98" in text
    assert "cnn_iid_validation_min_correct_prob >= 0.90" in text
    assert "selected_cnn_moon_moved_correct_prob >= 0.90" in text
    assert "selected_cnn_star_moved_correct_prob >= 0.90" in text
    assert "position-augmented train" in text
    assert "absolute object position" in text
    assert "Moons, Stars, and Clever Hans: What Did the Model Actually Learn?" in text
    assert "Accuracy says both models learned the task." in text
    assert "Unmasking Clever Hans predictors" in text
    assert "Sanity Checks for Saliency Maps" in text
    assert "fig00_apparent_shape_task.png" in text
    assert "fig00_model_cards.png" in text
    assert "fig01_both_models_appear_to_work.png" in text
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
    assert "Same moon. Same pixels. Different place. Different answer." in text
    assert "Same shape. Almost invisible background shift. Different belief." in text
    assert "The shortcut changes. The XAI discipline does not." in text
    assert (
        "Accuracy tells us whether the model was right on the exam. "
        "XAI asks whether it learned the rule we meant to teach."
    ) in text
    assert "Presentation mode: essential story only" in text
    assert "Use this as slide 5: Act I reveal." in text
    assert "00_hidden_position_shortcut.png" in text
    assert "01_same_shape_movement_counterfactual.png" in text
    assert "02_movement_confidence_paths.png" in text
    assert "03_position_response_maps_with_boundaries.png" in text
    assert "04_shape_position_score_surface.png" in text
    assert "05_why_heatmaps_are_not_enough.png" in text
    assert "06_shortcut_evidence_ledger.png" in text
    assert "fig_final_bridge_to_real_xai.png" in text
    assert "fig_final_xai_loop.png" in text
    assert "fig_final_real_world_bridge.png" in text
    assert "fig20_act2_apparent_shape_task_invisible_background.png" in text
    assert "fig20a_background_only_sanity_check.png" in text
    assert "fig21_act2_cnn_appears_to_work.png" in text
    assert "fig22_invisible_background_swap_counterfactual.png" in text
    assert "fig23_invisible_background_confidence_sweep.png" in text
    assert "fig24_invisible_background_difference_amplified.png" in text
    assert "fig25_background_tint_response_surface.png" in text
    assert "fig26_background_vs_shape_bars.png" in text
    assert "fig27_act2_heatmaps_are_not_enough.png" in text
    assert "fig28_act2_mitigation_retest.png" in text
    assert "fig29_two_act_evidence_board.png" in text
    assert "anim_invisible_background_morph_moon.gif" in text
    assert "anim_invisible_background_morph_moon.mp4" in text
    assert "anim_invisible_background_morph_star.gif" in text
    assert "anim_invisible_background_morph_star.mp4" in text
    assert "Act II: The CNN learns a different shortcut" in text
    assert "BackgroundMeanOnlyClassifier" in text
    assert "Act2CueCNN" in text
    assert "correct_class_probability" in text
    assert "margin_loss" in text
    assert "ACT2_TINT_DELTA_CANDIDATES" in text
    assert "make_invisible_background_dataset" in text
    assert "render_same_shape_with_background_style" in text
    assert 'act2_swap_scores["moon_on_moon_bg"] <= 0.05' in text
    assert 'act2_swap_scores["moon_on_star_bg"] >= 0.95' in text
    assert 'act2_swap_scores["star_on_star_bg"] >= 0.95' in text
    assert 'act2_swap_scores["star_on_moon_bg"] <= 0.05' in text
    assert 'act2_mitigated_swap_scores["moon_on_moon_bg"] <= 0.10' in text
    assert 'act2_mitigated_swap_scores["moon_on_star_bg"] <= 0.10' in text
    assert 'act2_mitigated_swap_scores["star_on_star_bg"] >= 0.90' in text
    assert 'act2_mitigated_swap_scores["star_on_moon_bg"] >= 0.90' in text
    assert "XAI is not a heatmap" in text
    assert "anim_moon_moves_confidence.gif" in text
    assert "anim_star_moves_confidence.gif" in text
    assert "anim_moon_moves_confidence.mp4" in text
    assert "anim_star_moves_confidence.mp4" in text
    assert "anim_response_map_path_mlp.gif" in text
    assert "anim_response_map_path_cnn.gif" in text
    assert "anim_morph_lower_left.gif" in text
    assert "anim_morph_upper_right.gif" in text
    assert "anim_moon_moves_heatmaps.gif" in text
    assert "anim_moon_moves_heatmaps.mp4" in text
    assert "anim_star_moves_heatmaps.gif" in text
    assert "anim_star_moves_heatmaps.mp4" in text
    assert "anim_morph_lower_left_heatmaps.gif" in text
    assert "anim_morph_lower_left_heatmaps.mp4" in text
    assert "anim_morph_upper_right_heatmaps.gif" in text
    assert "anim_morph_upper_right_heatmaps.mp4" in text
    assert "The heatmap overlays are deliberately secondary." in text
    assert "presentation_export_manifest" in text
    assert "movement_path_results" in text
    assert "shape_morph_results" in text
    assert "without using the sample tensor cache" in text
    assert 'shape_morph_results["lower_left"]["cnn"][-1] >= 0.90' in text
    assert 'shape_morph_results["upper_right"]["cnn"][-1] >= 0.90' in text
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
