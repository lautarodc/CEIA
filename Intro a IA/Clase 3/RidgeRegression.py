import numpy as np
from BaseModel import BaseModel


class RidgeRegression(BaseModel):

    def fit(self, X, y):
        self.model = mini_batch_gradient_descent(X, y)

    def predict(self, X):
        if len(X.shape) > 1:
            return X.dot(self.model)
        else:
            return X * self.model


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
        lamba = regularization
    """
    b = 16
    n = X_train.shape[0]
    l = 0

    # initialize random weights
    if len(X_train.shape) > 1:
        scalar = False
        m = X_train.shape[1]
        W = np.random.randn(m).reshape(m, 1)
    else:
        scalar = True
        m = 1
        W = np.random.randn(1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            if scalar:
                prediction = batch_X * W
            else:
                prediction = np.matmul(batch_X, W)  # nx1

            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2 / n * grad_sum  # 1xm

            if scalar:
                gradient = grad_mul
            else:
                gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W * (1 - 2 * lr * l) - (lr * gradient)

    return W
