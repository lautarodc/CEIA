import numpy as np
from BaseModel import BaseModel


class LinealRegressionAffine(BaseModel):

    def fit(self, x, y):
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.model = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)

    def predict(self, x):
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.matmul(x, self.model)
