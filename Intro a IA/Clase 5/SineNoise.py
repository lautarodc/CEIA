import numpy as np
import matplotlib.pyplot as plt
from metrics import MSE
from Gradients import gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent, k_folds
from LinealRegression import LinealRegression

MAX_ORDER = 15

# Data generation
amt_points = 360
x = np.linspace(0, 360, num=amt_points)
y = np.sin(x * np.pi / 180.)
noise = np.random.normal(0, .1, y.shape)
noisy_y = y + noise
plt.figure(1)
plt.plot(x, noisy_y)
plt.show()

# Split Data
percentage = 0.8
permuted_idxs = np.random.permutation(x.shape[0])
train_idxs = permuted_idxs[0:int(percentage * x.shape[0])]
test_idxs = permuted_idxs[int(percentage * x.shape[0]): x.shape[0]]
x_train = x[train_idxs]
x_test = x[test_idxs]
y_train = noisy_y[train_idxs]
y_test = noisy_y[test_idxs]
error = MSE()

# Model Initialization
lineal_regression = LinealRegression()

# Polynomic X train and X test
p_x_train = np.zeros((x_train.shape[0], MAX_ORDER + 1))
p_x_test = np.zeros((x_test.shape[0], MAX_ORDER + 1))
MSE_register = np.zeros((MAX_ORDER, 1))
Predictions = np.zeros((y_test.shape[0], MAX_ORDER))
MSE_notK = np.zeros((MAX_ORDER, 1))

p_x_train = np.hstack((np.ones((x_train.shape[0], 1)), p_x_train))
p_x_test = np.hstack((np.ones((x_test.shape[0], 1)), p_x_test))

for i in range(MAX_ORDER):
    p_x_train[:, i+1] = x_train ** (i + 1)
    p_x_test[:, i+1] = x_test ** (i + 1)
    MSE_register[i] = k_folds(p_x_train[:, :i + 2], y_train, 10)
    lineal_regression.fit(p_x_train[:, :i + 2], y_train)
    Predictions[:, i] = lineal_regression.predict(p_x_test[:, :i + 2])
    MSE_notK[i] = error(y_test, Predictions[:, i])
    ids = np.argsort(x_test, axis=0)
    plt.plot(x_test[ids], y_test[ids])
    plt.plot(x_test[ids], Predictions[:, i][ids])
    plt.show()


# Graphic
orders = np.arange(MAX_ORDER) + 1
plt.figure(2)
plt.plot(orders, MSE_register)
plt.show()

plt.figure(3)
plt.plot(orders, MSE_notK)
plt.show()

print(MSE_register)
