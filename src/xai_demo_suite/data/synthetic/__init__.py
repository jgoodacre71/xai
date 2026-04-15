"""Synthetic datasets for shortcuts, limits, and test fixtures."""

from xai_demo_suite.data.synthetic.fixtures import SyntheticImageSample, make_striped_fixture
from xai_demo_suite.data.synthetic.industrial_shortcuts import (
    IndustrialShortcutSample,
    generate_industrial_shortcut_dataset,
)
from xai_demo_suite.data.synthetic.slot_boards import SlotBoardSample, generate_slot_board_dataset

__all__ = [
    "IndustrialShortcutSample",
    "SlotBoardSample",
    "SyntheticImageSample",
    "generate_industrial_shortcut_dataset",
    "generate_slot_board_dataset",
    "make_striped_fixture",
]
