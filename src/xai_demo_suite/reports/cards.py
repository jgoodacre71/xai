"""Demo card artefacts for generated reports."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xai_demo_suite.reports.build_metadata import BuildMetadata
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
    build_metadata: BuildMetadata | None = None


KNOWN_PREPARED_DATASETS: tuple[tuple[str, str], ...] = (
    (
        "Waterbirds",
        "data/processed/waterbirds/waterbird_complete95_forest2water2/manifest.jsonl",
    ),
    (
        "MetaShift",
        "data/processed/metashift/subpopulation_shift_cat_dog_indoor_outdoor/manifest.jsonl",
    ),
    ("NEU-CLS", "data/processed/neu_cls/shortcut_binary/manifest.jsonl"),
    ("KolektorSDD2", "data/processed/ksdd2/shortcut_binary/manifest.jsonl"),
    ("MVTec AD bottle", "data/processed/mvtec_ad/bottle/manifest.jsonl"),
    ("MVTec LOCO AD juice_bottle", "data/processed/mvtec_loco_ad/juice_bottle/manifest.jsonl"),
    ("MVTec AD 2", "data/processed/mvtec_ad_2"),
    ("VisA", "data/processed/visa"),
)


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
    if card.build_metadata is not None and card.build_metadata.manifest_path is not None:
        metadata = dict(data["build_metadata"])
        metadata["manifest_path"] = _relative(card.build_metadata.manifest_path, root)
        data["build_metadata"] = metadata
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
    metadata_html = ""
    if card.build_metadata is not None:
        manifest_text = "Not used"
        if card.build_metadata.manifest_path is not None:
            manifest_text = _relative(card.build_metadata.manifest_path, root)
        metadata_html = (
            "<dt>Build metadata</dt><dd><ul>"
            f"<li>Git SHA: <code>{html.escape(card.build_metadata.git_sha)}</code></li>"
            f"<li>Built at: {html.escape(card.build_metadata.built_at_utc)}</li>"
            f"<li>Data mode: {html.escape(card.build_metadata.data_mode)}</li>"
            f"<li>Cache: {html.escape(card.build_metadata.cache_status)}</li>"
            f"<li>Manifest: <code>{html.escape(manifest_text)}</code></li>"
            "</ul></dd>"
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
    {metadata_html}
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
    project_root = root.parent
    sorted_cards = tuple(sorted(cards, key=_demo_sort_key))
    prepared_datasets = _prepared_dataset_status(project_root)
    prepared_count = sum(1 for _name, prepared in prepared_datasets if prepared)
    generated_reports = len(sorted_cards)
    dataset_badges = "".join(
        (
            f'<span class="badge {"ready" if prepared else "missing"}">'
            f"{html.escape(name)}: {'prepared' if prepared else 'missing'}"
            "</span>"
        )
        for name, prepared in prepared_datasets
    )
    ready_items = [name for name, prepared in prepared_datasets if prepared]
    ready_text = (
        ", ".join(ready_items)
        if ready_items
        else "No prepared external datasets detected."
    )
    items = []
    for card in sorted_cards:
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
        build_meta_html = ""
        if card.build_metadata is not None:
            build_meta_html = (
                '<div class="tile-build-meta">'
                f'<span class="pill">SHA {html.escape(card.build_metadata.git_sha)}</span>'
                f'<span class="pill">{html.escape(card.build_metadata.data_mode)}</span>'
                "</div>"
            )
        items.append(
            '<article class="demo-tile">'
            f'<a class="thumb" href="{html.escape(_relative(card.report_path, root))}">'
            f"{figure_html}</a>"
            '<div class="tile-body">'
            f"<h2>{html.escape(card.title)}</h2>"
            f"{build_meta_html}"
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
    .summary {{
      margin-top: 16px;
      display: grid;
      gap: 10px;
    }}
    .summary p {{
      margin: 0;
      color: #364152;
      font-size: 14px;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .badge {{
      font-size: 12px;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      color: #364152;
    }}
    .badge.ready {{
      background: #ecfdf3;
      border-color: #b7ebc6;
      color: #166534;
    }}
    .badge.missing {{
      background: #f8fafc;
      border-color: #d8dee4;
      color: #64748b;
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
    .tile-build-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0 0 10px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      background: #f8fafc;
      color: #364152;
      font-size: 12px;
      font-weight: 600;
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
      <div class="summary">
        <p>
          Reports generated: {generated_reports}. Prepared dataset adapters detected:
          {prepared_count} / {len(prepared_datasets)}.
        </p>
        <p>{html.escape(ready_text)}</p>
        <div class="badges">{dataset_badges}</div>
      </div>
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
    build_metadata = None
    raw_build_metadata = data.get("build_metadata")
    if isinstance(raw_build_metadata, dict):
        manifest_path = raw_build_metadata.get("manifest_path")
        build_metadata = BuildMetadata(
            git_sha=str(raw_build_metadata["git_sha"]),
            built_at_utc=str(raw_build_metadata["built_at_utc"]),
            data_mode=str(raw_build_metadata["data_mode"]),
            manifest_path=root / str(manifest_path) if isinstance(manifest_path, str) else None,
            cache_status=str(raw_build_metadata["cache_status"]),
        )
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
        build_metadata=build_metadata,
    )


def save_demo_index_for_output_root(output_root: Path) -> Path:
    """Write an index containing all demo cards under an output root."""

    cards: list[DemoCard] = []
    for card_path in sorted(output_root.glob("*/demo_card.json")):
        data = json.loads(card_path.read_text(encoding="utf-8"))
        cards.append(_card_from_json_dict(data, output_root))
    return save_demo_index(tuple(cards), output_root / "index.html")


def _demo_sort_key(card: DemoCard) -> tuple[int, str]:
    match = re.search(r"Demo (\d+)", card.title)
    if match is None:
        return (999, card.title)
    return (int(match.group(1)), card.title)


def _prepared_dataset_status(project_root: Path) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (name, (project_root / relative_path).exists())
        for name, relative_path in KNOWN_PREPARED_DATASETS
    )
