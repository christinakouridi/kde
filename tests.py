""""
Simple tests on the dataset. Usage:
python -m pytest tests.py
"""

import numpy as np
from src.data import get_dataset


def test_mnist_range():
    data = get_dataset('mnist')
    for x in data:
        assert np.min(x) >= 0 and np.max(x) <= 1


def test_cifar_range():
    data = get_dataset('cifar-100')
    for x in data:
        assert np.min(x) >= 0 and np.max(x) <= 1


def test_mnist_length():
    data = get_dataset('cifar-100')
    for x in data:
        assert len(x) == 10_000


def test_cifar_length():
    data = get_dataset('cifar-100')
    for x in data:
        assert len(x) == 10_000