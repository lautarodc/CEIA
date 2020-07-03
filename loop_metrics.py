from precision_IA import PrecisionIA
from recall_IA import RecallIA
from accuracy_IA import AccuracyIA
from average_query_precision_IA import AverageQueryPrecision
from k_average_query_precision_IA import AverageQueryAtK


class LoopMetrics(object):
    def __init__(self, **kwargs):
        self.data = kwargs
        self.metrics = {}

    def get_metrics(self):
        metrics_options = [PrecisionIA, RecallIA, AccuracyIA, AverageQueryPrecision, AverageQueryAtK]
        for metric in metrics_options:
            aux = metric(**self.data)
            self.metrics[metric.__name__] = aux()
        return self.metrics
