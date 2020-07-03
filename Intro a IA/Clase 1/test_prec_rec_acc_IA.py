import numpy as np
import pytest
from prec_rec_acc_IA import MetricsIA


@pytest.fixture(scope='module')
def example_array():
    return MetricsIA(np.array([1, 1, 0, 1, 0, 1, 0, 0]), np.array([0, 1, 1, 0, 1, 1, 0, 1]))


def test_precision(example_array):
    assert example_array.get_precision() == 0.5


def test_recall(example_array):
    assert example_array.get_recall() == 0.4


def test_accuracy(example_array):
    assert example_array.get_accuracy() == 0.375
