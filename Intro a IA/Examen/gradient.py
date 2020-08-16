import numpy as np


class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


def mini_batch_gradient_descent(X_train, y_train, valid, y_valid, lr=0.001, amt_epochs=100):
    b = 30

    # initialize random weights
    if len(X_train.shape) > 1:
        scalar = False
        m = X_train.shape[1]
        W = np.random.randn(m).reshape(m, 1)
    else:
        scalar = True
        m = 1
        W = np.random.randn(1)

    # Almacenamiento de errores de train y validation
    mse_train = []
    mse_valid = []
    error_valid = MSE()

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        error_batch = 0
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]
            batch_y = batch_y.reshape(-1, 1)

            if scalar:
                prediction = batch_X * W
            else:
                prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1
            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -(2 / batch_size) * grad_sum  # 1xm

            if scalar:
                gradient = grad_mul
            else:
                gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            error_batch = error_batch + error_valid(batch_y, prediction)
            W = W - (lr * gradient)

        # Calculo de errores de entrenamiento y validación
        mse_train.append(error_batch / b)
        prediction_valid = valid.dot(W)
        mse_valid.append(error_valid(y_valid.reshape(-1, 1), prediction_valid))

    return W, mse_train, mse_valid
