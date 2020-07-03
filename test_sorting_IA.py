import numpy as np
from sorting_IA import sorting_vector_l2


def test_sorting():
    arrange = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    result = sorting_vector_l2(arrange)
    assert np.all(expected == result)