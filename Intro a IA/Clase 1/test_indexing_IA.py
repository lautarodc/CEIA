import numpy as np
from indexing_IA import IndexerVector


def test_indexing():
    arrange = IndexerVector(np.array([1, 15, 3, 2, 4, 1, 15, 8], dtype=np.int64))
    idx, valid_idx = arrange.get_idx(np.array([15, 3, 5]))
    assert np.all(valid_idx == np.array([True, True, False]))
    ids = arrange.get_id(idx)
    assert np.all(ids == np.array([15, 3]))
