import numpy as np
from BaseModel import BaseModel


class LinealRegressionAffine(BaseModel):

    def fit(self, X, y):
        X = np.vstack((X, np.ones(len(X)))).T
        W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        X = np.vstack((X, np.ones(len(X)))).T
        return X.dot(self.model)
