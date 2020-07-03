import numpy as np
import pytest
from precision_IA import PrecisionIA
from recall_IA import RecallIA
from accuracy_IA import AccuracyIA
from average_query_precision_IA import AverageQueryPrecision
from k_average_query_precision_IA import AverageQueryAtK
from loop_metrics import LoopMetrics


@pytest.fixture(scope='module')
def example_data():
    predictions = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    truth = np.array([0, 1, 1, 0, 1, 1, 0, 1])
    q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    pred_rank = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
    truth_relev = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    k = 3
    data = {"predictions": predictions, "truth": truth, "q_id": q_id, "predicted_rank": pred_rank,
            "truth_relevance": truth_relev, "k": k}
    return data


def test_individual(example_data):
    data = example_data
    metrics_options = [PrecisionIA, RecallIA, AccuracyIA, AverageQueryPrecision, AverageQueryAtK]
    validations = [0.5, 0.4, 0.375, 0.5, 0.5]
    for i, metric in enumerate(metrics_options):
        aux = metric(**data)
        print(aux())
        assert aux() == validations[i]


def test_loop_metrics(example_data):
    data = example_data
    looper = LoopMetrics(**data)
    assert looper.get_metrics() == {"PrecisionIA": 0.5, "RecallIA": 0.4, "AccuracyIA": 0.375,
                                    "AverageQueryPrecision": 0.5,
                                    "AverageQueryAtK": 0.5}
