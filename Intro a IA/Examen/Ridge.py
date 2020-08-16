import numpy as np
from BaseModel import BaseModel


class RidgeRegression(BaseModel):

    def fit(self, X, y, ld):
        self.model = mini_batch_gradient_descent(X, y, ld=ld)

    def predict(self, X):
        if len(X.shape) > 1:
            return X.dot(self.model)
        else:
            return X * self.model


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100, ld=0):
    b = 10
    n = X_train.shape[0]

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
        y_train = y_train.reshape(-1, 1)

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

            W = W * (1 - 2 * lr * ld) - (lr * gradient)

    return W


class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


def k_folds_ridge(X_train, y_train, k=5, ld=0):
    l_regression = RidgeRegression()
    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list = []
    model_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])
        l_regression.fit(new_X_train, new_y_train, ld)
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid.reshape(-1, 1), prediction))
        model_list.append(l_regression.model)

    mean_MSE = np.mean(mse_list)
    idx = np.argmin(mse_list)
    models = np.array(model_list)
    # Se toma como mejor modelo aquel que tiene menor MSE sobre el set de validación.

    return mean_MSE, models[idx]
