import numpy as np
# import matplotlib.pyplot as plt
from BaseModel import BaseModel


class NNXor(BaseModel):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y, lr=0.05, b=16, epochs=3000):
        # Random initialization of parameters
        W1 = np.random.uniform(-1, 1, (2, 3))
        W2 = np.random.uniform(-1, 1, (3, 2))
        W3 = np.random.uniform(-1, 1, (2, 1))
        b1 = np.random.uniform(-1, 1, (3, 1))
        b2 = np.random.uniform(-1, 1, (2, 1))
        b3 = np.random.uniform(-1, 1, (1, 1))
        print(W1)
        epoch_loss = []
        epoch_count = []

        for j in range(epochs):
            idx = np.random.permutation(x.shape[0])
            x_train = x[idx]
            y_train = y[idx]
            batch_size = int(len(x_train) / b)

            for i in range(0, len(x_train), batch_size):
                end = i + batch_size if i + batch_size <= len(x_train) else len(x_train)
                batch_x = x_train[i: end].T
                batch_y = y_train[i: end].T


                # Feed forward
                Z1 = W1.T @ batch_x + b1  # R3xb
                A1 = self.sigmoid(Z1)  # R3xb
                Z2 = W2.T @ A1 + b2  # R2xb
                A2 = self.sigmoid(Z2)  # R2xb
                Z3 = W3.T @ A2 + b3  # R1xb
                y_pred = self.sigmoid(Z3)  # R1xb
                error = np.mean((batch_y - y_pred) ** 2)
                print(error)
                # Backprop
                dz3 = -2 * error * self.sigmoid(Z3) * (1 - self.sigmoid(Z3))  # R1xb
                w3_grad = (1 / batch_size) * (dz3 @ A2.T)  # R1x2
                b3_grad = (1 / batch_size) * np.sum(dz3, axis=1, keepdims=True)  # R1x1
                dz2 = np.multiply(W3 @ dz3, self.sigmoid(Z2) * (1 - self.sigmoid(Z2)))  # R2xb
                w2_grad = (1 / batch_size) * (dz2 @ A1.T)  # R1x2
                b2_grad = (1 / batch_size) * np.sum(dz2, axis=1, keepdims=True)  # R2x1
                dz1 = np.multiply(W2 @ dz2, self.sigmoid(Z1) * (1 - self.sigmoid(Z1)))  # R3xb
                w1_grad = (1 / batch_size) * (dz1 @ batch_x.T)  # R3x2
                b1_grad = (1 / batch_size) * np.sum(dz1, axis=1, keepdims=True)  # R3x1

                # Update
                W3 = W3 - lr * w3_grad.T
                b3 = b3 - lr * b3_grad
                W2 = W2 - lr * w2_grad.T
                b2 = b2 - lr * b2_grad
                W1 = W1 - lr * w1_grad.T
                b1 = b1 - lr * b1_grad
            epoch_count.append(j)
            epoch_loss.append(error)
            print(y_pred, batch_y)
            print(error)

        self.model = [W1, b1, W2, b2, W3, b3]
        return epoch_count, epoch_loss

    def predict(self, x):
        # Feed forward
        W1 = self.model[0]
        b1 = self.model[1]
        W2 = self.model[2]
        b2 = self.model[3]
        W3 = self.model[4]
        b3 = self.model[5]

        Z1 = W1.T @ x + b1  # R3xb
        A1 = self.sigmoid(Z1)  # R3xb
        Z2 = W2.T @ A1 + b2  # R2xb
        A2 = self.sigmoid(Z2)  # R2xb
        Z3 = W3.T @ A2 + b3  # R1xb
        y_pred = self.sigmoid(Z3)  # R1xb
        return y_pred

    def get_model(self):
        return self.model


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('x_1', np.float),
                     ('x_2', np.float),
                     ('y', np.float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def get_data(self):
        X = np.array([self.dataset['x_1'], self.dataset['x_2']]).T
        y = self.dataset['y']
        return X, y


if __name__ == '__main__':
    dataset_train = Data('clase_2_train_data.csv')
    x_train, y_train = dataset_train.get_data()
    dataset_test = Data('clase_2_test_data.csv')
    x_test, y_test = dataset_test.get_data()
    NN = NNXor()
    e_count, e_loss = NN.fit(x_train, y_train, 0.01, 10, 10000)
    predictions = NN.predict(x_test.T)
    print(y_test, predictions.T)
