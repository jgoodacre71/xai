"""Synthetic datasets for shortcuts, limits, and test fixtures."""

from xai_demo_suite.data.synthetic.fixtures import SyntheticImageSample, make_striped_fixture
from xai_demo_suite.data.synthetic.slot_boards import SlotBoardSample, generate_slot_board_dataset

__all__ = [
    "SlotBoardSample",
    "SyntheticImageSample",
    "generate_slot_board_dataset",
    "make_striped_fixture",
]
