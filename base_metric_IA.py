class BaseMetric(object):
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def __call__(self, *args, **kwargs):
        return "Base class for a Metric"
