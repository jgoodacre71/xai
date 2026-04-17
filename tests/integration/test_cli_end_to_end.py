from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    repo_root = _repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_demo_cli_suite_verify_and_review_pack(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    suite = _run_module(
        "xai_demo_suite.cli.demo",
        "suite",
        "--output-root",
        str(output_root),
        "--no-cache",
    )
    assert suite.returncode == 0, suite.stderr or suite.stdout
    assert "built: waterbirds-shortcut" in suite.stdout
    assert (output_root / "index.html").exists()

    verify = _run_module(
        "xai_demo_suite.cli.demo",
        "verify",
        "--output-root",
        str(output_root),
    )
    assert verify.returncode == 0, verify.stderr or verify.stdout
    assert "verification: ok" in verify.stdout

    review_pack = _run_module(
        "xai_demo_suite.cli.demo",
        "review-pack",
        "--output-root",
        str(output_root),
        "--output-dir",
        str(output_root / "review_pack"),
    )
    assert review_pack.returncode == 0, review_pack.stderr or review_pack.stdout
    assert (output_root / "review_pack" / "index.html").exists()


def test_data_cli_list_reports_supported_datasets() -> None:
    result = _run_module("xai_demo_suite.cli.data", "list")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "MVTec AD" in result.stdout
    assert "Waterbirds" in result.stdout
    assert "KolektorSDD2" in result.stdout
