import numpy as np
from BaseModel import BaseModel


class LinealRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        # return np.matmul(X, self.model)
        return X.dot(self.model)
        # return self.model * X
