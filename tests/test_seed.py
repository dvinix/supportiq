"""Tests for random seed reproducibility."""

import os
import random

from supportiq.core.seed import set_seed


def test_set_seed_reproducibility():
    set_seed(123)
    val1 = random.random()
    val2 = random.randint(1, 1000)

    set_seed(123)
    val3 = random.random()
    val4 = random.randint(1, 1000)

    assert val1 == val3
    assert val2 == val4
    assert os.environ.get("PYTHONHASHSEED") == "123"
