import numpy as np
import matplotlib.pyplot as plt
from Ridge import RidgeRegression, k_folds_ridge
from gradient import mini_batch_gradient_descent


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


def z_score_dataset(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


if __name__ == '__main__':
    MAX_ORDER = 4

    dataset = Data('clase_8_dataset.csv')
    plt.figure(1)
    plt.scatter(dataset.dataset['x'], dataset.dataset['y'], label="Dataset")
    plt.title("Distribución de puntos")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper left")
    plt.show()
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
    plt.figure(2)
    plt.title("Prediccion sobre test data con mejor polinomio")
    plt.scatter(X_test, y_test, label="Test Data")
    plt.scatter(X_test, best_prediction, label="Predicción fórmula cerrada")
    plt.text(0, 150, "Orden: {order}".format(order=best_order + 1))
    plt.legend(loc="upper left")
    plt.show()

    print("MSE Fórmula Cerrada sobre test: {MSE} ".format(MSE=error(y_test, best_prediction)))

    # Split entre train y validation para mini batch. Normalizacion para evitar divergencia
    X = p_x_train[:, :best_order + 2]
    X[:, 1:] = z_score_dataset(X[:, 1:])
    valid_size = int(len(X) / 5)
    permute_ids = np.random.permutation(X.shape[0])
    valid_ids = permute_ids[0:valid_size]
    train_ids = permute_ids[valid_size:]
    valid = X[valid_ids]
    train = X[train_ids]
    y_train_split = y_train[train_ids]
    y_valid = y_train[valid_ids]
    test = p_x_test[:, :best_order + 2]
    test[:, 1:] = z_score_dataset(test[:, 1:])

    # Parametros mini batch
    EPOCHS = 100
    lr = 1E-3

    model_mb, mse_train_mb, mse_valid_mb = mini_batch_gradient_descent(train, y_train_split, valid, y_valid, lr, EPOCHS)

    # Grafica error de train y validation mini batch
    array_epochs = np.arange(EPOCHS)
    predict_mb = test.dot(model_mb)
    plt.figure(3)
    plt.scatter(X_test, y_test, label="Test Data")
    plt.scatter(X_test, predict_mb, label="Predicción Mini Batch")
    plt.legend(loc="upper left")
    plt.show()

    print("MSE Mini batch sobre test: {MSE} ".format(MSE=error(y_test.reshape(-1, 1), predict_mb)))

    plt.figure(4)
    plt.title("Error de train y validation mini batch vs. epochs")
    plt.plot(array_epochs, mse_train_mb, label="MSE Train")
    plt.plot(array_epochs, mse_valid_mb, label="MSE Validation")
    plt.legend(loc="upper left")
    plt.show()

    # Regresión con Ridge - vector de lambdas para K-folds
    lambdas = np.linspace(0, 2, 100)
    mse_lambda = []
    rigde_model = []
    for i, ld in enumerate(lambdas):
        mse, model = k_folds_ridge(X, y_train, 5, ld)
        mse_lambda.append(mse)
        rigde_model.append(model)

    min_ridge = np.argmin(mse_lambda)
    rigde_model = np.array(rigde_model)
    best_ridge_model = rigde_model[min_ridge]
    ridge_prediction = test.dot(best_ridge_model)
    mse_ridge = error(ridge_prediction, y_test.reshape(-1, 1))
    lambdas = np.array(lambdas)
    print("MSE Ridge Regression: {MSE}, mejor lambda: {ld}".format(MSE=mse_ridge, ld=lambdas[min_ridge]))
    plt.figure(5)
    plt.scatter(X_test, y_test, label= "Test data")
    plt.scatter(X_test, ridge_prediction, label ="Predicción Ridge")
    plt.legend(loc="upper left")
    plt.show()


