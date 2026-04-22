from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


def _load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _execute_notebook(path: Path) -> None:
    notebook = _load_notebook(path)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        exec(compile(source, str(path), "exec"), namespace)


@pytest.mark.parametrize(
    "notebook_rel_path",
    (
        "overview/00_overview.ipynb",
        "shortcut_lab/01_waterbirds_shortcut.ipynb",
        "shortcut_lab/02_industrial_shortcut_trap.ipynb",
        "patchcore_explainability/03_patchcore_mvtec_ad.ipynb",
        "patchcore_explainability/04_patchcore_wrong_normal.ipynb",
        "patchcore_limits/05_patchcore_count_limit.ipynb",
        "patchcore_limits/06_patchcore_severity_limit.ipynb",
        "patchcore_limits/07_patchcore_loco_logic_limit.ipynb",
        "robustness_drift/08_explanation_drift.ipynb",
    ),
)
def test_notebook_smoke_execution(
    monkeypatch: pytest.MonkeyPatch,
    notebook_rel_path: str,
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    cwd_before = Path.cwd()
    _execute_notebook(Path("notebooks") / notebook_rel_path)
    assert Path.cwd() == cwd_before
    assert os.environ["MPLBACKEND"] == "Agg"
