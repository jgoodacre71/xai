"""Demo card artefacts for generated reports."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xai_demo_suite.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class DemoCard:
    """Short summary card for a demo release artefact."""

    title: str
    task: str
    model: str
    explanation_methods: tuple[str, ...]
    key_lesson: str
    failure_mode: str
    intervention: str
    remaining_caveats: tuple[str, ...]
    report_path: Path
    figure_paths: tuple[Path, ...]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _card_to_json_dict(card: DemoCard, root: Path) -> dict[str, object]:
    data = asdict(card)
    data["report_path"] = _relative(card.report_path, root)
    data["figure_paths"] = [_relative(path, root) for path in card.figure_paths]
    data["explanation_methods"] = list(card.explanation_methods)
    data["remaining_caveats"] = list(card.remaining_caveats)
    return data


def save_demo_card(card: DemoCard, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and HTML demo card artefacts."""

    ensure_directory(output_dir)
    json_path = output_dir / "demo_card.json"
    html_path = output_dir / "demo_card.html"
    root = output_dir.parent
    json_path.write_text(
        json.dumps(_card_to_json_dict(card, root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    methods = "".join(f"<li>{html.escape(method)}</li>" for method in card.explanation_methods)
    caveats = "".join(f"<li>{html.escape(caveat)}</li>" for caveat in card.remaining_caveats)
    figures = "".join(
        f'<li><code>{html.escape(_relative(path, root))}</code></li>' for path in card.figure_paths
    )
    report_href = html.escape(_relative(card.report_path, output_dir))
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(card.title)} Demo Card</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
    }}
    main {{ max-width: 880px; margin: 0 auto; }}
    dt {{ font-weight: 700; margin-top: 16px; }}
    dd {{ margin: 6px 0 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(card.title)}</h1>
  <dl>
    <dt>Task</dt><dd>{html.escape(card.task)}</dd>
    <dt>Model</dt><dd>{html.escape(card.model)}</dd>
    <dt>Explanation methods</dt><dd><ul>{methods}</ul></dd>
    <dt>Key lesson</dt><dd>{html.escape(card.key_lesson)}</dd>
    <dt>Failure mode</dt><dd>{html.escape(card.failure_mode)}</dd>
    <dt>Intervention</dt><dd>{html.escape(card.intervention)}</dd>
    <dt>Remaining caveats</dt><dd><ul>{caveats}</ul></dd>
    <dt>Report</dt><dd><a href="{report_href}">Open report</a></dd>
    <dt>Selected figures</dt><dd><ul>{figures}</ul></dd>
  </dl>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return json_path, html_path


def save_demo_index(cards: tuple[DemoCard, ...], output_path: Path) -> Path:
    """Write a local index page for generated demo artefacts."""

    ensure_directory(output_path.parent)
    root = output_path.parent
    items = []
    for card in cards:
        card_path = card.report_path.parent / "demo_card.html"
        figure_path = card.figure_paths[0] if card.figure_paths else None
        figure_html = (
            f'<img src="{html.escape(_relative(figure_path, root))}" '
            f'alt="{html.escape(card.title)} selected figure">'
            if figure_path is not None
            else ""
        )
        methods = "".join(
            f"<li>{html.escape(method)}</li>" for method in card.explanation_methods[:3]
        )
        caveat = card.remaining_caveats[0] if card.remaining_caveats else "See report."
        items.append(
            '<article class="demo-tile">'
            f'<a class="thumb" href="{html.escape(_relative(card.report_path, root))}">'
            f"{figure_html}</a>"
            '<div class="tile-body">'
            f"<h2>{html.escape(card.title)}</h2>"
            f"<p>{html.escape(card.key_lesson)}</p>"
            f"<dl>"
            f"<dt>Model</dt><dd>{html.escape(card.model)}</dd>"
            f"<dt>Intervention</dt><dd>{html.escape(card.intervention)}</dd>"
            f"<dt>Caveat</dt><dd>{html.escape(caveat)}</dd>"
            f"</dl>"
            f"<ul class=\"methods\">{methods}</ul>"
            '<div class="links">'
            f"<a href=\"{html.escape(_relative(card.report_path, root))}\">Open report</a>"
            f"<a href=\"{html.escape(_relative(card_path, root))}\">Demo card</a>"
            "</div>"
            "</div>"
            "</article>"
        )
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XAI Demo Suite Local Reports</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      color: #20252d;
      background: #f5f7fa;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 24px 48px;
    }}
    header {{
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.1;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      max-width: 760px;
      color: #52606d;
      font-size: 16px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .demo-tile {{
      display: grid;
      grid-template-rows: 190px auto;
      overflow: hidden;
      border: 1px solid #d8dee4;
      border-radius: 8px;
      background: #ffffff;
    }}
    .thumb {{
      display: block;
      background: #e6ebf0;
    }}
    .thumb img {{
      width: 100%;
      height: 190px;
      object-fit: cover;
      display: block;
    }}
    .tile-body {{
      padding: 16px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 19px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 12px;
      line-height: 1.45;
    }}
    dl {{
      display: grid;
      grid-template-columns: 88px 1fr;
      gap: 6px 10px;
      margin: 0 0 12px;
      font-size: 13px;
    }}
    dt {{
      font-weight: 700;
      color: #364152;
    }}
    dd {{
      margin: 0;
      color: #52606d;
    }}
    .methods {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      list-style: none;
      margin: 0 0 14px;
      padding: 0;
      font-size: 12px;
    }}
    .methods li {{
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 4px 8px;
      background: #f8fafc;
    }}
    .links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .links a {{
      color: #0b5cab;
      font-weight: 700;
      text-decoration: none;
    }}
    .links a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>XAI Demo Suite Local Reports</h1>
      <p>
        Static local entry point for the generated explainability demos. Each
        tile links to the full report and its concise demo card.
      </p>
    </header>
    <section class="grid">
      {''.join(items)}
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def _card_from_json_dict(data: dict[str, Any], root: Path) -> DemoCard:
    return DemoCard(
        title=str(data["title"]),
        task=str(data["task"]),
        model=str(data["model"]),
        explanation_methods=tuple(str(item) for item in data["explanation_methods"]),
        key_lesson=str(data["key_lesson"]),
        failure_mode=str(data["failure_mode"]),
        intervention=str(data["intervention"]),
        remaining_caveats=tuple(str(item) for item in data["remaining_caveats"]),
        report_path=root / str(data["report_path"]),
        figure_paths=tuple(root / str(path) for path in data["figure_paths"]),
    )


def save_demo_index_for_output_root(output_root: Path) -> Path:
    """Write an index containing all demo cards under an output root."""

    cards: list[DemoCard] = []
    for card_path in sorted(output_root.glob("*/demo_card.json")):
        data = json.loads(card_path.read_text(encoding="utf-8"))
        cards.append(_card_from_json_dict(data, output_root))
    return save_demo_index(tuple(cards), output_root / "index.html")
