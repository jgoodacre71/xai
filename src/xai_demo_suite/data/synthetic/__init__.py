"""Synthetic datasets for shortcuts, limits, and test fixtures."""

from xai_demo_suite.data.synthetic.fixtures import SyntheticImageSample, make_striped_fixture
from xai_demo_suite.data.synthetic.industrial_shortcuts import (
    IndustrialShortcutSample,
    generate_industrial_shortcut_dataset,
)
from xai_demo_suite.data.synthetic.slot_boards import (
    NuisanceBoardSample,
    SlotBoardSample,
    generate_nuisance_board_dataset,
    generate_severity_sweep_dataset,
    generate_slot_board_dataset,
)
from xai_demo_suite.data.synthetic.waterbirds import (
    HabitatBirdSample,
    generate_habitat_bird_dataset,
    render_habitat_bird_image,
    write_habitat_counterfactual,
)

__all__ = [
    "HabitatBirdSample",
    "IndustrialShortcutSample",
    "NuisanceBoardSample",
    "SlotBoardSample",
    "SyntheticImageSample",
    "generate_habitat_bird_dataset",
    "generate_industrial_shortcut_dataset",
    "generate_nuisance_board_dataset",
    "generate_severity_sweep_dataset",
    "generate_slot_board_dataset",
    "make_striped_fixture",
    "render_habitat_bird_image",
    "write_habitat_counterfactual",
]
