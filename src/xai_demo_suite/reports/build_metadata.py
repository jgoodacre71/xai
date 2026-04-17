"""Build metadata helpers for generated reports."""

from __future__ import annotations

import html
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BuildMetadata:
    """Small build stamp attached to reports and demo cards."""

    git_sha: str
    built_at_utc: str
    data_mode: str
    manifest_path: Path | None
    cache_status: str


@lru_cache(maxsize=1)
def current_git_sha() -> str:
    """Return the current short git SHA, or ``unknown`` if unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def make_build_metadata(
    *,
    data_mode: str,
    manifest_path: Path | None = None,
    cache_enabled: bool | None = None,
) -> BuildMetadata:
    """Create a build metadata stamp for one report build."""

    cache_status = "not applicable"
    if cache_enabled is True:
        cache_status = "enabled"
    elif cache_enabled is False:
        cache_status = "disabled"
    return BuildMetadata(
        git_sha=current_git_sha(),
        built_at_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        data_mode=data_mode,
        manifest_path=manifest_path,
        cache_status=cache_status,
    )


def render_build_metadata_section(metadata: BuildMetadata) -> str:
    """Render a simple build-metadata section for non-flagship reports."""

    manifest_text = "Not used"
    if metadata.manifest_path is not None:
        manifest_text = metadata.manifest_path.as_posix()
    return f"""
  <section>
    <h2>Build Metadata</h2>
    <ul>
      <li>Git SHA: <code>{html.escape(metadata.git_sha)}</code></li>
      <li>Built at: {html.escape(metadata.built_at_utc)}</li>
      <li>Data mode: {html.escape(metadata.data_mode)}</li>
      <li>Cache: {html.escape(metadata.cache_status)}</li>
      <li>Manifest: <code>{html.escape(manifest_text)}</code></li>
    </ul>
  </section>
"""
