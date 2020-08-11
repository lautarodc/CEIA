import numpy as np
from BaseModel import BaseModel
from metrics import Accuracy, Precision, Recall


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(BaseModel):

    def fit(self, X, y, lr, b, epochs):

        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # initialize random weights
        m = X.shape[1]
        W = np.random.randn(m).reshape(m, 1)

        for j in range(epochs):
            idx = np.random.permutation(X.shape[0])
            X_train = X[idx]
            y_train = y[idx]
            batch_size = int(len(X_train) / b)

            for i in range(0, len(X_train), batch_size):
                end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
                batch_X = X_train[i: end]
                batch_y = y_train[i: end]

                # prediction = sigmoid(batch_X @ W)
                # error = prediction - batch_y
                # grad_sum = np.sum(batch_X.T @ error, axis=0)
                # gradient = (1 / batch_size) * grad_sum

                # Implementacion Eze
                prediction = sigmoid(np.sum(np.transpose(W) * batch_X, axis=1))
                error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)
                grad_sum = np.sum(error * batch_X, axis=0)
                grad_mul = 1 / b * grad_sum
                gradient = np.transpose(grad_mul).reshape(-1, 1)

                W = W - (lr * gradient)
        self.model = W

    def predict(self, X):
        # mu = np.mean(X, axis=0)
        # std = np.std(X, axis=0)
        # X = (X - mu) / std
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        p = sigmoid(X @ self.model)
        mask_true = p >= 0.5
        mask_false = p < 0.5
        p[mask_true] = 1
        p[mask_false] = 0
        return p


def k_folds(X_train, y_train, lr, b, epochs, k=5):
    l_regression = LogisticRegression()
    error = Accuracy()

    chunk_size = int(len(X_train) / k)
    mse_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train, lr, b, epochs)
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))

    mean_MSE = np.mean(mse_list)

    return mean_MSE
