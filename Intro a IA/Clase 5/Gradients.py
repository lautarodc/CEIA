import numpy as np
from LinealRegression import LinealRegression
from metrics import MSE


def gradient_descent(x_train, y_train, lr=0.01, amt_epochs=100):
    # Initialize w
    w = np.random.randn(1)
    for i in range(amt_epochs):
        j = (y_train - x_train * w) * x_train
        j_sum = np.sum(j, axis=0)
        j_sum = j_sum * (-2 / x_train.shape[0])
        w = w - lr * j_sum
    return w


def stochastic_gradient_descent(x_train, y_train, lr=0.01, amt_epochs=100):
    # Initialize w
    w = np.random.randn(1)
    for i in range(amt_epochs):
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]
        for k in range(x_train.shape[0]):
            j = (y_train[k] - x_train[k] * w) * x_train[k]
            j_sum = j * (-2 / x_train.shape[0])
            w = w - lr * j_sum
    return w


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
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

            W = W - (lr * gradient)

    return W


def k_folds(X_train, y_train, k=5):
    l_regression = LinealRegression()
    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))

    mean_MSE = np.mean(mse_list)

    return mean_MSE
