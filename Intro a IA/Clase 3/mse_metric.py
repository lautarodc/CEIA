import numpy as np
from base_metric_IA import BaseMetric


class MseMetric(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        self.predictions = self.parameters["predictions"]
        self.truth = self.parameters["truth"]
        return np.sum((self.truth - self.predictions) ** 2, axis=0) / self.truth.shape[0]
