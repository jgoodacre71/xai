from __future__ import annotations

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script_name", "expected_report"),
    (
        ("01_waterbirds_shortcut.py", "waterbirds_shortcut/index.html"),
        ("02_industrial_shortcut_trap.py", "shortcut_industrial/index.html"),
        ("04_patchcore_wrong_normal.py", "patchcore_wrong_normal/index.html"),
        ("08_explanation_drift.py", "explanation_drift/index.html"),
    ),
)
def test_notebook_script_smoke_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    expected_report: str,
) -> None:
    monkeypatch.setenv("XAI_DEMO_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("XAI_DEMO_NOTEBOOK_OUTPUT_ROOT", str(tmp_path / "outputs"))

    runpy.run_path(str(Path("notebooks") / script_name), run_name="__main__")

    assert (tmp_path / "outputs" / expected_report).exists()
