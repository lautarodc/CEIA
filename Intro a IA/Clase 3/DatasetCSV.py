import numpy as np


class SplitData(object):

    def __init__(self, file_name):
        self.data = self._build_dataset(file_name)
        self.train_x, self.test_x, self.train_y, self.test_y = self.split(0.8)

    def _build_dataset(self, file_name):
        structure = [('income', np.float),
                     ('happiness', np.float)]

        with open(file_name, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[1]), float(line.split(',')[2]))  # add here + 10 in second value
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):
        x = self.data['income']
        y = self.data['happiness']

        permuted_idxs = np.random.permutation(x.shape[0])
        train_idxs = permuted_idxs[0:int(percentage * x.shape[0])]
        test_idxs = permuted_idxs[int(percentage * x.shape[0]): x.shape[0]]

        x_train = x[train_idxs]
        x_test = x[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return x_train, x_test, y_train, y_test

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

    def get_data(self):
        return self.data
