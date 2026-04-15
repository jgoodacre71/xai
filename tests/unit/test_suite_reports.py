from __future__ import annotations

from pathlib import Path

from xai_demo_suite.reports.suite import build_demo_suite, verify_demo_suite_outputs


def test_build_demo_suite_writes_synthetic_reports_and_verifies(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    results = build_demo_suite(output_root=output_root, include_mvtec=False, use_cache=False)
    verification = verify_demo_suite_outputs(output_root)

    assert {result.name for result in results} == {
        "waterbirds-shortcut",
        "shortcut-industrial",
        "patchcore-limits",
        "patchcore-wrong-normal",
        "explanation-drift",
    }
    assert all(result.status == "built" for result in results)
    assert verification.ok
    assert verification.card_count == 5
    assert (output_root / "index.html").exists()


def test_verify_demo_suite_outputs_reports_missing_cards(tmp_path: Path) -> None:
    result = verify_demo_suite_outputs(tmp_path / "outputs")

    assert not result.ok
    assert result.card_count == 0
    assert any("No demo cards found" in problem for problem in result.problems)
