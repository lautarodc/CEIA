import numpy as np
from base_metric_IA import BaseMetric


class AverageQueryPrecision(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        count_unique_id = np.bincount(self.parameters["q_id"])
        count_truth_relevance = np.bincount(self.parameters["q_id"] * self.parameters["truth_relevance"])
        return np.mean(count_truth_relevance[1::] / count_unique_id[1::])
