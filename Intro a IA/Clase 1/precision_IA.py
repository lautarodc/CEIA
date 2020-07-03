import numpy as np
from base_metric_IA import BaseMetric


class PrecisionIA(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        self.predictions = self.parameters["predictions"]
        self.truth = self.parameters["truth"]
        inverse_truth = np.where(self.truth == 0, 1, 0)
        inverse_predictions = np.where(self.predictions == 0, 1, 0)
        self.TP = np.sum(self.predictions * self.truth)
        self.TN = np.sum(inverse_predictions * inverse_truth)
        self.FN = np.sum(self.truth * inverse_predictions)
        self.FP = np.sum(self.predictions * inverse_truth)
        return self.TP / (self.TP + self.FP)
