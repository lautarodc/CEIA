
class BaseModel(object):
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented
