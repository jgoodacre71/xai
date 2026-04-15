"""Static report builders for demos."""

from xai_demo_suite.reports.cards import DemoCard, save_demo_card, save_demo_index
from xai_demo_suite.reports.explanation_drift import (
    ExplanationDriftReportConfig,
    build_explanation_drift_report,
)
from xai_demo_suite.reports.patchcore_bottle import (
    PatchCoreBottleReportConfig,
    build_patchcore_bottle_report,
)
from xai_demo_suite.reports.patchcore_limits import (
    PatchCoreLimitsReportConfig,
    build_patchcore_limits_report,
)
from xai_demo_suite.reports.patchcore_wrong_normal import (
    PatchCoreWrongNormalReportConfig,
    build_patchcore_wrong_normal_report,
)
from xai_demo_suite.reports.shortcut_industrial import (
    IndustrialShortcutReportConfig,
    build_industrial_shortcut_report,
)
from xai_demo_suite.reports.suite import (
    SuiteBuildResult,
    SuiteVerificationResult,
    build_demo_suite,
    verify_demo_suite_outputs,
)

__all__ = [
    "DemoCard",
    "ExplanationDriftReportConfig",
    "IndustrialShortcutReportConfig",
    "PatchCoreBottleReportConfig",
    "PatchCoreLimitsReportConfig",
    "PatchCoreWrongNormalReportConfig",
    "SuiteBuildResult",
    "SuiteVerificationResult",
    "build_demo_suite",
    "build_explanation_drift_report",
    "build_industrial_shortcut_report",
    "build_patchcore_bottle_report",
    "build_patchcore_limits_report",
    "build_patchcore_wrong_normal_report",
    "save_demo_card",
    "save_demo_index",
    "verify_demo_suite_outputs",
]
