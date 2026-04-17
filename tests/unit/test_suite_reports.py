from __future__ import annotations

from pathlib import Path

import pytest

import xai_demo_suite.reports.suite as suite_module
from xai_demo_suite.reports.explanation_drift import ExplanationDriftReportConfig
from xai_demo_suite.reports.patchcore_bottle import PatchCoreBottleReportConfig
from xai_demo_suite.reports.patchcore_logic import PatchCoreLogicReportConfig
from xai_demo_suite.reports.shortcut_industrial import IndustrialShortcutReportConfig
from xai_demo_suite.reports.suite import build_demo_suite, verify_demo_suite_outputs
from xai_demo_suite.reports.waterbirds_shortcut import WaterbirdsShortcutReportConfig


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


def test_build_demo_suite_disables_implicit_local_data_for_temp_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waterbirds_configs: list[WaterbirdsShortcutReportConfig] = []
    industrial_configs: list[IndustrialShortcutReportConfig] = []
    drift_configs: list[ExplanationDriftReportConfig] = []
    logic_configs: list[PatchCoreLogicReportConfig] = []

    def fake_waterbirds_report(config: WaterbirdsShortcutReportConfig) -> Path:
        waterbirds_configs.append(config)
        output_path = config.output_dir / "index.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    def fake_drift_report(config: ExplanationDriftReportConfig) -> Path:
        drift_configs.append(config)
        output_path = config.output_dir / "index.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    def fake_industrial_report(config: IndustrialShortcutReportConfig) -> Path:
        industrial_configs.append(config)
        output_path = config.output_dir / "index.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    def fake_logic_report(config: PatchCoreLogicReportConfig) -> Path:
        logic_configs.append(config)
        output_path = config.output_dir / "index.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    monkeypatch.setattr(suite_module, "build_waterbirds_shortcut_report", fake_waterbirds_report)
    monkeypatch.setattr(suite_module, "build_industrial_shortcut_report", fake_industrial_report)
    monkeypatch.setattr(suite_module, "build_explanation_drift_report", fake_drift_report)
    monkeypatch.setattr(suite_module, "build_patchcore_logic_report", fake_logic_report)

    build_demo_suite(output_root=tmp_path / "outputs", include_mvtec=False, use_cache=False)

    assert len(waterbirds_configs) == 1
    assert not waterbirds_configs[0].use_real_data

    assert len(industrial_configs) == 1
    assert not industrial_configs[0].use_real_data

    assert len(drift_configs) == 1
    assert not drift_configs[0].include_mvtec_if_available
    assert not drift_configs[0].industrial_manifest_path.exists()

    assert len(logic_configs) == 1
    assert not logic_configs[0].manifest_path.exists()
    assert "_disabled/patchcore_logic/manifest.jsonl" in logic_configs[0].manifest_path.as_posix()


def test_verify_demo_suite_outputs_reports_missing_cards(tmp_path: Path) -> None:
    result = verify_demo_suite_outputs(tmp_path / "outputs")

    assert not result.ok
    assert result.card_count == 0
    assert any("No demo cards found" in problem for problem in result.problems)


def test_verify_demo_suite_outputs_reports_missing_semantic_markers(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report_dir = output_root / "waterbirds_shortcut"
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "index.html").write_text("XAI Demo Suite Local Reports", encoding="utf-8")
    (report_dir / "index.html").write_text("<html><body>stub</body></html>", encoding="utf-8")
    (report_dir / "demo_card.html").write_text("<html><body>stub</body></html>", encoding="utf-8")
    (report_dir / "figure.png").write_bytes(b"png")
    (report_dir / "demo_card.json").write_text(
        """
{
  "title": "Demo 01 - Waterbirds Shortcut",
  "task": "task",
  "model": "model",
  "explanation_methods": ["Grad-CAM"],
  "key_lesson": "lesson",
  "failure_mode": "failure",
  "intervention": "intervention",
  "remaining_caveats": ["caveat"],
  "report_path": "waterbirds_shortcut/index.html",
  "figure_paths": ["waterbirds_shortcut/figure.png"]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = verify_demo_suite_outputs(output_root)

    assert not result.ok
    assert any("Missing marker" in problem for problem in result.problems)


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
        mvtec_benchmark_limit=40,
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
    assert config.max_benchmark_records == 40
    assert not config.use_cache
