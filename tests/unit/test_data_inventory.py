from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_ieee_candidate_register_loads() -> None:
    register_path = Path("data/ieee_candidates.yaml")
    data = yaml.safe_load(register_path.read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    assert data["candidates"] == []
    required_fields = data["schema"]["required_fields"]
    assert "access_type" in required_fields
    assert "work_permission_status" in required_fields
    assert set(data["schema"]["fit_categories"]) == {"A", "B", "C", "D"}


def test_data_inventory_script_runs(tmp_path: Path) -> None:
    output_path = tmp_path / "inventory.txt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_data_inventory.py",
            "--root",
            ".",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
    report = output_path.read_text(encoding="utf-8")
    assert "Generated Moons/Stars Clever-Hans Demo" in report
    assert "Waterbirds" in report
    assert "IEEE DataPort Candidate Register" in report
    assert "This report records local file presence only" in report
