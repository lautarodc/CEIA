import numpy as np
import math as math
import pytest
from norm_IA import norm_vector_l0, norm_vector_l1, norm_vector_l2, norm_vector_inf


@pytest.fixture(scope='module')
def example_matrix():
    return np.array([[1, 0, 0, 4], [5, 6, 0, 8]])


def test_norm_l0(example_matrix):
    arrange = example_matrix
    expected = np.array([2, 3])
    result = norm_vector_l0(arrange)
    assert np.all(expected == result)


def test_norm_l1(example_matrix):
    arrange = example_matrix
    expected = np.array([5, 19])
    result = norm_vector_l1(arrange)
    assert np.all(expected == result)


def test_norm_l2(example_matrix):
    arrange = example_matrix
    expected = np.array([math.sqrt(17), math.sqrt(125)])
    result = norm_vector_l2(arrange)
    assert np.all(expected == result)


def test_norm_inf(example_matrix):
    arrange = example_matrix
    expected = np.array([4, 8])
    result = norm_vector_inf(arrange)
    assert np.all(expected == result)