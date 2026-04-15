"""Static report builders for demos."""

from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index
from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    build_patchcore_bottle_report,
)

__all__ = [
    "DemoCard",
    "PatchCoreBottleReportConfig",
    "build_patchcore_bottle_report",
    "save_demo_card",
    "save_demo_index",
]
