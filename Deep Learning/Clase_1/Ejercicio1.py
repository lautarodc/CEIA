import numpy as np
from BaseModel import BaseModel


class NNXor(BaseModel):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x_train, y_train, lr=0.05, amt_epochs=3000):
        w_out = np.random.uniform(0, 1, (2, 1))
        b_out = np.random.uniform(0, 1, 1)
        w_hidden = np.random.uniform(0, 1, (2, 2))
        b_hidden = np.random.uniform(0, 1, (2, 1))
        epoch_loss = []
        epoch_count = []
        threshold = 0.5
        for i in range(amt_epochs):
            idx = np.random.permutation(x_train.shape[0])
            x_train = x_train[idx]
            y_train = y_train[idx]
            for k in range(4):
                # Feed forward
                z1 = x_train[k, 0] * w_hidden[0, 0] + x_train[k, 1] * w_hidden[0, 1] + b_hidden[0]
                z2 = x_train[k, 0] * w_hidden[1, 0] + x_train[k, 1] * w_hidden[1, 1] + b_hidden[1]
                a1 = self.sigmoid(z1)
                a2 = self.sigmoid(z2)
                prediction = a1 * w_out[0] + a2 * w_out[1] + b_out
                error = (y_train[k] - prediction)**2
                print(x_train[k, 0], x_train[k, 1], y_train[k], prediction)

                # Backprop
                w_out[0] = w_out[0] + 2 * lr * error * a1
                w_out[1] = w_out[1] + 2 * lr * error * a2
                b_out = b_out + 2 * lr * error
                sigma1 = a1 * (1 - a1)
                sigma2 = a2 * (1 - a2)
                w_hidden[0, 0] = w_hidden[0, 0] + 2 * lr * error * w_out[0] * x_train[k, 0] * sigma1
                w_hidden[0, 1] = w_hidden[0, 1] + 2 * lr * error * w_out[0] * x_train[k, 1] * sigma1
                w_hidden[1, 0] = w_hidden[1, 0] + 2 * lr * error * w_out[1] * x_train[k, 0] * sigma2
                w_hidden[1, 1] = w_hidden[1, 1] + 2 * lr * error * w_out[1] * x_train[k, 1] * sigma2
                b_hidden[0] = b_hidden[0] + 2 * lr * error * w_out[0] * sigma1
                b_hidden[1] = b_hidden[1] + 2 * lr * error * w_out[1] * sigma2

            epoch_count.append(i)
            epoch_loss.append(error)
        self.model = [w_out, b_out, w_hidden, b_hidden]
        return epoch_count, epoch_loss

    def predict(self, x):
        z = self.model[2].dot(x.T.reshape(-1, 1)) + self.model[3]
        a = self.sigmoid(z)
        prediction = self.model[0][:, 0].dot(a[:, 0]) + self.model[1].reshape(-1, 1)
        print(prediction.shape)
        return prediction


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print(x)
    y = np.array([0, 1, 1, 0])
    NN_nonvectorized = NNXor()
    epoch_n, epoch_loss = NN_nonvectorized.fit(x, y, 0.01, 1000)
    y_predicted = NN_nonvectorized.predict(x[1, :])
    print(y_predicted)
