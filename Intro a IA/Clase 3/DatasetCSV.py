import numpy as np


class SplitData(object):
    instance = None

    def __init__(self, file_name):
        self.data = self._build_dataset(file_name)
        x = self.data['income'][:, np.newaxis]
        y = self.data['fitness'][:, np.newaxis]
        self.train_x, self.valid_x, self.test_x = self.split(x)
        self.train_y, self.valid_y, self.test_y = self.split(y)

    def _build_dataset(self, file_name):
        self.structured_type = np.dtype(
            [('id', np.int64), ('income', np.float32), ('fitness', np.float32)])
        return np.genfromtxt(file_name, dtype=self.structured_type, delimiter=",", skip_header=1)

    def split(self, data):
        train_percent = 0.7
        valid_percent = 0.2
        train_index = int(train_percent*data.shape[0])
        valid_index = int(valid_percent*data.shape[0])
        splitted = np.split(data, [train_index, train_index+valid_index])
        return splitted[0], splitted[1], splitted[2]

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_valid_data(self):
        return self.valid_x, self.valid_y

    def get_test_data(self):
        return self.test_x, self.test_y

    def get_data(self):
        return self.data

