import numpy as np
import matplotlib.pyplot as plt
from DatasetCSV import SplitData
from LinealRegression import LinealRegression
from LinealRegressionAffine import LinealRegressionAffine
from metrics import MSE

MAX_ORDER = 50
LINEAL_TYPE = 'AFFINE'

# Data extraction and preprocessing
data_example = SplitData('income.csv')
x_train, y_train = data_example.get_train_data()
x_validation, y_validation = data_example.get_test_data()
y_train = y_train
y_validation = y_validation

# Model Initialization
lineal_regression = LinealRegression()
affine_regression = LinealRegression()

# Metrics
MSE_register = np.zeros((MAX_ORDER, 1))
Predictions = np.zeros((y_validation.shape[0], MAX_ORDER))
mse = MSE()

# Polynomic X train and X test
p_x_train = np.zeros((x_train.shape[0], MAX_ORDER))
p_x_test = np.zeros((x_validation.shape[0], MAX_ORDER))

for i in range(MAX_ORDER):
    p_x_train[:, i] = x_train ** (i + 1)
    p_x_test[:, i] = x_validation ** (i + 1)

    if LINEAL_TYPE == 'CLASSIC':
        lineal_regression.fit(p_x_train[:, :i + 1], y_train)
        Predictions[:, i] = lineal_regression.predict(p_x_test[:, :i + 1])
        MSE_register[i] = mse(y_validation, Predictions[:, i])
    else:
        affine_regression.fit(np.hstack((p_x_train[:, :i + 1], np.ones((x_train.shape[0], 1))))
                              , y_train)
        Predictions[:, i] = affine_regression.predict(
            np.hstack((p_x_test[:, :i + 1], np.ones((x_validation.shape[0], 1))))
            )
        MSE_register[i] = mse(y_validation, Predictions[:, i])

# Graphics

font = {
    'family': 'DejaVu sans',
    'color': 'black',
    'weight': 'normal',
    'size': 9
}

grades = np.arange(MAX_ORDER) + 1

plt.figure(1)
plt.scatter(grades, MSE_register)
plt.grid(True)
plt.title("MSE for different Orders ")
plt.xlabel("Order")
plt.ylabel("MSE")

plt.figure(2)

plt.subplots_adjust(hspace=0.5)

plt.subplot(411)
plt.grid(True)
plt.title("Lineal Regression")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.text(12, 5, 'Mean Square Error = %s' % MSE_register[0], fontdict=font)
plt.scatter(x_validation, y_validation)
plt.plot(x_validation, Predictions[:, 0])

plt.subplot(412)
plt.grid(True)
plt.title("Cubic Regression")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.text(12, 5, 'Mean Square Error = %s' % MSE_register[2], fontdict=font)
plt.scatter(x_validation, y_validation)
plt.plot(x_validation, Predictions[:, 2])

plt.subplot(413)
plt.grid(True)
plt.title("Sixth Order Regression")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.text(12, 5, 'Mean Square Error = %s' % MSE_register[5], fontdict=font)
plt.scatter(x_validation, y_validation)
plt.plot(x_validation, Predictions[:, 5])

plt.subplot(414)
plt.grid(True)
plt.title("Tenth Order Regression")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.text(12, 5, 'Mean Square Error = %s' % MSE_register[9], fontdict=font)
plt.scatter(x_validation, y_validation)
plt.plot(x_validation, Predictions[:, 9])


plt.show()

print(MSE_register)
