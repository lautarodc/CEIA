import numpy as np
import pytest
from average_query_precision_IA import AverageQueryPrecision
from k_average_query_precision_IA import AverageQueryAtK


@pytest.fixture(scope='module')
def example_query():
    q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    pred_rank = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
    truth_relev = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    k = 3
    return AverageQueryPrecision(q_id, pred_rank, truth_relev), AverageQueryAtK(q_id, pred_rank, truth_relev, k)


def test_average_query_precision(example_query):
    aqp, aqp_k = example_query
    assert aqp.get_avg_query_precision() == 0.5
    assert aqp_k.get_avg_query_precision() == 0.5
