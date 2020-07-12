import numpy as np
from BaseModel import BaseModel


class LinealRegression(BaseModel):

    def fit(self, x, y):
        self.model = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)

    def predict(self, x):
        return np.matmul(x, self.model)
