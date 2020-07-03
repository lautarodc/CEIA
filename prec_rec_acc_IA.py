import numpy as np


class MetricsIA(object):
    def __init__(self, predictions, truth):
        self.predictions = predictions
        self.truth = truth
        inverse_truth = np.where(self.truth == 0, 1, 0)
        inverse_predictions = np.where(self.predictions == 0, 1, 0)
        self.TP = np.sum(self.predictions * self.truth)
        self.TN = np.sum(inverse_predictions * inverse_truth)
        self.FN = np.sum(self.truth * inverse_predictions)
        self.FP = np.sum(self.predictions * inverse_truth)

    def get_precision(self):
        return self.TP / (self.TP + self.FP)

    def get_recall(self):
        return self.TP / (self.TP + self.FN)

    def get_accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
