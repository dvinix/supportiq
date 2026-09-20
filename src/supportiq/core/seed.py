"""Reproducibility utilities for setting random seeds across libraries."""

import os
import random


def set_seed(seed: int = 42) -> None:
    """Set random seed across random, numpy, and torch (if installed) for reproducibility.

    Args:
        seed: The integer seed to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
