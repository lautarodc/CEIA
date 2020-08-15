import numpy as np
# import matplotlib.pyplot as plt
from Ridge import RidgeRegression


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('x', np.float),
                     ('y', np.float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]))

                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):
        X = self.dataset['x']
        y = self.dataset['y']

        permuted_idxs = np.random.permutation(X.shape[0])

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test


class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


class BaseModel(object):
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented


class LinealRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return X.dot(self.model)


def k_folds(X_train, y_train, k=5):
    l_regression = LinealRegression()
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
        l_regression.fit(new_X_train, new_y_train)
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))
        model_list.append(l_regression.model)

    mean_MSE = np.mean(mse_list)
    idx = np.argmin(mse_list)
    models = np.array(model_list)
    # Se toma como mejor modelo aquel que tiene menor MSE sobre el set de validación.

    return mean_MSE, models[idx]


def mini_batch_gradient_descent(X_train, y_train, valid, y_valid, lr=0.01, amt_epochs=100):
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
        for j in range(0, len(X_train), batch_size):
            end = j + batch_size if j + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[j: end]
            batch_y = y_train[j: end]
            batch_y = batch_y.reshape(-1, 1)

            if scalar:
                prediction = batch_X * W
            else:
                prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1
            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2 / batch_size * grad_sum  # 1xm

            if scalar:
                gradient = grad_mul
            else:
                gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            error_batch = error_batch + error_valid(batch_y, prediction)
            W = W - (lr * gradient)

        # Calculo de errores de entrenamiento y validación
        mse_train.append(error_batch/b)
        prediction_valid = np.matmul(valid, W)
        mse_valid.append(error_valid(y_valid, prediction_valid))

    return W, mse_train, mse_valid


if __name__ == '__main__':
    MAX_ORDER = 4

    dataset = Data('clase_8_dataset.csv')
    # plt.figure(1)
    # plt.scatter(dataset.dataset['x'], dataset.dataset['y'])
    # plt.title("Distribución de puntos")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()
    X_train, X_test, y_train, y_test = dataset.split(0.8)
    error = MSE()
    p_x_train = np.zeros((X_train.shape[0], MAX_ORDER + 1))
    p_x_test = np.zeros((X_test.shape[0], MAX_ORDER + 1))
    MSE_validation = np.zeros((MAX_ORDER, 1))
    MSE_test = np.zeros_like(MSE_validation)
    model_list = []
    p_x_train = np.hstack((np.ones((X_train.shape[0], 1)), p_x_train))
    p_x_test = np.hstack((np.ones((X_test.shape[0], 1)), p_x_test))

    for i in range(MAX_ORDER):
        p_x_train[:, i + 1] = X_train ** (i + 1)
        p_x_test[:, i + 1] = X_test ** (i + 1)
        MSE_validation[i], model = k_folds(p_x_train[:, :i + 2], y_train, 5)
        model_list.append(model)
        prediction = p_x_test[:, :i + 2].dot(model)
        MSE_test[i] = error(y_test, prediction)

    # Tomamos como mejor polinomio el que tiene menor MSE sobre el conjunto de test
    best_order = np.argmin(MSE_test)
    best_model = model_list[best_order]
    print(best_model)
    best_prediction = p_x_test[:, :best_order + 2].dot(best_model)
    # plt.figure(2)
    # plt.title("Prediccion sobre data de test con mejor polinomio")
    # plt.scatter(X_test, y_test)
    # plt.scatter(X_test, best_prediction)
    # plt.show()
    # plt.text(0, 150, "Orden: {order}".format(order=best_order))

    # Split entre train y validation
    X = p_x_train[:, :2]
    valid_size = int(len(X) / 5)
    permute_ids = np.random.permutation(X.shape[0])
    valid_ids = permute_ids[0:valid_size]
    train_ids = permute_ids[valid_size:]
    valid = X[valid_ids]
    train = X[train_ids]
    y_train_split = y_train[train_ids]
    y_valid = y_train[valid_ids]

    # Parametros mini batch
    EPOCHS = 200
    lr = 0.001

    model_mb, mse_train_mb, mse_valid_mb = mini_batch_gradient_descent(train, y_train_split, valid, y_valid, lr, EPOCHS)

    # Grafica error de train y validation mini batch
    array_epochs = np.arange(EPOCHS)

    # plt.figure(3)
    # plt.title("Error de train y validation mini batch vs. epochs")
    # plt.plot(array_epochs, mse_train_mb)
    # plt.plot(array_epochs, mse_valid_mb)
