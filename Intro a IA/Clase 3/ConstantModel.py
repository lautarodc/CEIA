import numpy as np
from BaseModel import BaseModel


class ConstantModel(BaseModel):

    def fit(self, x, y):
        self.model = np.mean(y, axis=0)

    def predict(self, x):
        return self.model
