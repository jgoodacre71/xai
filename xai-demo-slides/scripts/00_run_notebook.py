from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "xai_demo.ipynb"
ASSETS_DIR = ROOT / "assets"
REPORT = ROOT / "deck" / "build_report.md"


def append_report(text: str) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Build Report\n\n"
    REPORT.write_text(existing.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        allow_errors=args.allow_errors,
        resources={"metadata": {"path": str(ROOT)}},
    )

    try:
        client.execute()
    except Exception as exc:
        tb = traceback.format_exc()[:1000]
        append_report(
            "# Notebook execution failed\n\n"
            f"- notebook: `{NOTEBOOK.name}`\n"
            f"- error: `{type(exc).__name__}`\n\n"
            "```text\n"
            f"{tb}\n"
            "```"
        )
        raise

    executed_path = ASSETS_DIR / "xai_demo_executed.ipynb"
    nbformat.write(nb, executed_path)
    append_report(
        "# Notebook execution succeeded\n\n"
        f"- source: `{NOTEBOOK}`\n"
        f"- executed copy: `{executed_path}`"
    )


if __name__ == "__main__":
    main()
