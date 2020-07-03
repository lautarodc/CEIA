import numpy as np
from base_metric_IA import BaseMetric


class AverageQueryAtK(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        q_id = self.parameters["q_id"]
        predicted_rank = self.parameters["predicted_rank"]
        truth_relevance = self.parameters["truth_relevance"]
        k = self.parameters["k"]
        count_unique_id = np.bincount(q_id[predicted_rank < k])
        count_truth_relevance = np.bincount(q_id[predicted_rank < k] * truth_relevance[predicted_rank < k])
        return np.mean(np.round(count_truth_relevance[1::] / count_unique_id[1::], decimals=1))
