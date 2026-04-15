from __future__ import annotations

from pathlib import Path

import pytest

import xai_demo_suite.reports.suite as suite_module
from xai_demo_suite.reports.patchcore_bottle import PatchCoreBottleReportConfig
from xai_demo_suite.reports.suite import build_demo_suite, verify_demo_suite_outputs


def test_build_demo_suite_writes_synthetic_reports_and_verifies(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    results = build_demo_suite(output_root=output_root, include_mvtec=False, use_cache=False)
    verification = verify_demo_suite_outputs(output_root)

    assert {result.name for result in results} == {
        "waterbirds-shortcut",
        "shortcut-industrial",
        "patchcore-limits",
        "patchcore-severity",
        "patchcore-logic",
        "patchcore-wrong-normal",
        "explanation-drift",
    }
    assert all(result.status == "built" for result in results)
    assert verification.ok
    assert verification.card_count == 7
    assert (output_root / "index.html").exists()


def test_verify_demo_suite_outputs_reports_missing_cards(tmp_path: Path) -> None:
    result = verify_demo_suite_outputs(tmp_path / "outputs")

    assert not result.ok
    assert result.card_count == 0
    assert any("No demo cards found" in problem for problem in result.problems)


def test_suite_passes_mvtec_model_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[PatchCoreBottleReportConfig] = []

    def fake_bottle_report(config: PatchCoreBottleReportConfig) -> Path:
        captured.append(config)
        output_path = config.output_dir / "index.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    monkeypatch.setattr(suite_module, "build_patchcore_bottle_report", fake_bottle_report)

    results = build_demo_suite(
        output_root=tmp_path / "outputs",
        include_mvtec=True,
        use_cache=False,
        mvtec_manifest_path=tmp_path / "manifest.jsonl",
        mvtec_feature_extractor_name="feature_map_resnet18_pretrained",
        mvtec_max_train=20,
        mvtec_max_examples=3,
        mvtec_coreset_size=512,
        mvtec_input_size=224,
    )

    assert any(result.name == "patchcore-bottle" for result in results)
    assert len(captured) == 1
    config = captured[0]
    assert config.manifest_path == tmp_path / "manifest.jsonl"
    assert config.feature_extractor_name == "feature_map_resnet18_pretrained"
    assert config.max_train == 20
    assert config.max_examples == 3
    assert config.coreset_size == 512
    assert config.input_size == 224
    assert not config.use_cache
