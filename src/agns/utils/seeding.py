from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and the hash randomization for reproducibility.

    Parameters
    ----------
    seed:
        The seed value applied to all sources of randomness used in this project
        (no PyTorch dependency here).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
