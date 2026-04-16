from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebooks_are_output_free() -> None:
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    assert notebooks

    for notebook_path in notebooks:
        notebook = _load_notebook(notebook_path)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell.get("execution_count") is None, notebook_path
            assert cell.get("outputs") == [], notebook_path


def test_required_demo_notebooks_exist() -> None:
    notebooks = {path.name for path in Path("notebooks").glob("*.ipynb")}

    assert {
        "00_overview.ipynb",
        "01_waterbirds_shortcut.ipynb",
        "02_industrial_shortcut_trap.ipynb",
        "03_patchcore_mvtec_ad.ipynb",
        "04_patchcore_wrong_normal.ipynb",
        "05_patchcore_count_limit.ipynb",
        "06_patchcore_severity_limit.ipynb",
        "07_patchcore_loco_logic_limit.ipynb",
        "08_explanation_drift.ipynb",
    }.issubset(notebooks)


def test_required_demo_notebook_scripts_exist() -> None:
    notebook_scripts = {path.name for path in Path("notebooks").glob("*.py")}

    assert {
        "00_overview.py",
        "01_waterbirds_shortcut.py",
        "02_industrial_shortcut_trap.py",
        "03_patchcore_mvtec_ad.py",
        "04_patchcore_wrong_normal.py",
        "05_patchcore_count_limit.py",
        "06_patchcore_severity_limit.py",
        "07_patchcore_loco_logic_limit.py",
        "08_explanation_drift.py",
    }.issubset(notebook_scripts)


def test_notebook_scripts_follow_shared_narrative_template() -> None:
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

    for script_path in sorted(Path("notebooks").glob("0*.py")):
        text = script_path.read_text(encoding="utf-8")
        for section in required_sections:
            assert section in text, script_path


def test_demo_notebooks_use_package_report_builders() -> None:
    expectations = {
        "01_waterbirds_shortcut.ipynb": (
            "from xai_demo_suite.reports.waterbirds_shortcut import",
            "build_waterbirds_shortcut_report(config)",
        ),
        "02_industrial_shortcut_trap.ipynb": (
            "from xai_demo_suite.reports.shortcut_industrial import",
            "build_industrial_shortcut_report(config)",
        ),
        "03_patchcore_mvtec_ad.ipynb": (
            "from xai_demo_suite.reports.patchcore_bottle import",
            "build_patchcore_bottle_report(config)",
        ),
        "04_patchcore_wrong_normal.ipynb": (
            "from xai_demo_suite.reports.patchcore_wrong_normal import",
            "build_patchcore_wrong_normal_report(config)",
        ),
        "05_patchcore_count_limit.ipynb": (
            "from xai_demo_suite.reports.patchcore_limits import",
            "build_patchcore_limits_report(config)",
        ),
        "06_patchcore_severity_limit.ipynb": (
            "from xai_demo_suite.reports.patchcore_severity import",
            "build_patchcore_severity_report(config)",
        ),
        "07_patchcore_loco_logic_limit.ipynb": (
            "from xai_demo_suite.reports.patchcore_logic import",
            "build_patchcore_logic_report(config)",
        ),
        "08_explanation_drift.ipynb": (
            "from xai_demo_suite.reports.explanation_drift import",
            "build_explanation_drift_report(config)",
        ),
    }

    for notebook_name, expected_strings in expectations.items():
        notebook = _load_notebook(Path("notebooks") / notebook_name)
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for expected in expected_strings:
            assert expected in code
