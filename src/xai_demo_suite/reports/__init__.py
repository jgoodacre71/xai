"""Static report builders for demos."""

from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index
from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    build_patchcore_bottle_report,
)
from xai_demo_suite.reports.patchcore_limits import (
    PatchCoreLimitsReportConfig,
    build_patchcore_limits_report,
)
from xai_demo_suite.reports.shortcut_industrial import (
    IndustrialShortcutReportConfig,
    build_industrial_shortcut_report,
)

__all__ = [
    "DemoCard",
    "IndustrialShortcutReportConfig",
    "PatchCoreBottleReportConfig",
    "PatchCoreLimitsReportConfig",
    "build_industrial_shortcut_report",
    "build_patchcore_bottle_report",
    "build_patchcore_limits_report",
    "save_demo_card",
    "save_demo_index",
]
