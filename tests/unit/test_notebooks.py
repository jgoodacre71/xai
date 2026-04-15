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


def test_patchcore_notebook_uses_package_report_builder() -> None:
    notebook = _load_notebook(Path("notebooks/03_patchcore_mvtec_bottle.ipynb"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "from xai_demo_suite.reports.patchcore_bottle import" in code
    assert "build_patchcore_bottle_report(config)" in code
    assert "PatchCoreBottleReportConfig" in code
