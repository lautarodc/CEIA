import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from LogisticRegression import k_folds
# from sklearn.linear_model import LogisticRegression
from metrics import Accuracy, Precision, Recall


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('x_1', np.float),
                     ('x_2', np.float),
                     ('y', np.float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]))
                        # add here + 10 in second value
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):  # 0.8
        X = np.array([self.dataset['x_1'], self.dataset['x_2']]).T
        y = self.dataset['y']

        permuted_idxs = np.random.permutation(X.shape[0])

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    dataset = Data('clase_6_dataset.txt')
    # Normalization
    # dataset.dataset['x_1'] = (dataset.dataset['x_1'] - np.mean(dataset.dataset['x_1'], axis=0)) / np.std(
    #     dataset.dataset['x_1'], axis=0)
    # dataset.dataset['x_2'] = (dataset.dataset['x_2'] - np.mean(dataset.dataset['x_2'], axis=0)) / np.std(
    #     dataset.dataset['x_2'], axis=0)
    X_train, X_test, y_train, y_test = dataset.split(0.8)

    # Scikit-learn implementation comparision
    # logisticRegr = LogisticRegression(penalty='none', fit_intercept=False, intercept_scaling=0)
    # logisticRegr.fit(X_train, y_train)
    # predictions = logisticRegr.predict(X_test)
    # score = logisticRegr.score(X_test, y_test)
    # print(score)
    # print(logisticRegr.coef_)
    # print(logisticRegr.intercept_)

    # K-Folds for best learning rate determination
    lr_list = np.linspace(0.001, 0.1, 100)
    kfolds_lr = np.zeros(lr_list.shape)
    for i, lr in enumerate(lr_list):
        kfolds_lr[i] = k_folds(X_train, y_train.reshape(-1, 1), lr, 10, 100)
    best_lr = lr_list[np.argmax(kfolds_lr)]

    # K-Folds for best batch size determination
    batch_list = np.linspace(1, 30, 30)
    kfolds_b = np.zeros(batch_list.shape)
    for i, b in enumerate(batch_list):
        kfolds_b[i] = k_folds(X_train, y_train.reshape(-1, 1), best_lr, b, 100)
    best_b = batch_list[np.argmax(kfolds_b)]

    # Fit model and predict with optimized parameters
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train.reshape(-1, 1), best_lr, best_b, 50000)
    print(logistic_regression.model)
    predictions = logistic_regression.predict(X_test)
    slope = -(logistic_regression.model[1] / logistic_regression.model[2])
    intercept = -(logistic_regression.model[0] / logistic_regression.model[2])

    # Metrics
    metrics = [Precision(), Accuracy(), Recall()]
    for metric in metrics:
        print('{metric}: {value}'.format(metric=metric.__class__.__name__, value=metric(y_test, predictions[:, 0])))

    # Graphics
    plt.figure(1)
    plt.scatter(dataset.dataset['x_1'], dataset.dataset['x_2'], c=dataset.dataset['y'])
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax = plt.gca()
    y_vals = intercept + (slope * dataset.dataset['x_1'])
    ax.autoscale(False)
    plt.plot(dataset.dataset['x_1'], y_vals, c="k")
    plt.show()

    f, (ax, bx) = plt.subplots(2, 1, sharey='col')

    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    ax.set_title('Valores reales')

    bx.scatter(X_test[:, 0], X_test[:, 1], c=predictions[:, 0])
    y_vals = intercept + (slope * X_test[:, 0])
    bx.plot(X_test[:, 0], y_vals, c="k")
    bx.set_title('Predicciones')

    plt.show()
