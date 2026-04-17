"""Seed helpers for reproducible demos and tests."""

from __future__ import annotations

import os
import random
from contextlib import suppress

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and hash randomisation, then return a generator."""

    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
