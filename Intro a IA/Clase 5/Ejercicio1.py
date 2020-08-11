import numpy as np
import matplotlib.pyplot as plt
from metrics import MSE
from DatasetCSV import SplitData
from Gradients import gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent
from LinealRegression import LinealRegression
from RidgeRegression import RidgeRegression

# Data extraction and preprocessing
data_example = SplitData('../Clase 4/income.csv')
x_train, y_train = data_example.get_train_data()
x_test, y_test = data_example.get_test_data()

# Prediction
# w_grad = mini_batch_gradient_descent(x_train, y_train, 0.01, 100)
# print(w_grad)
# y_predicted = x_test*w_grad
ridge = RidgeRegression()
ridge.fit(x_train, y_train)
y_predicted = ridge.predict(x_test)

error = MSE()
lineal_regression = LinealRegression()
lineal_regression.fit(x_train, y_train)
print(lineal_regression.model)
y_predicted_regression = x_test*lineal_regression.model
print(error(y_test, y_predicted))
print(error(y_test, y_predicted_regression))


plt.figure(1)
plt.subplot(311)
plt.grid(True)
plt.title("Validation Data")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.scatter(x_test, y_test)

plt.subplot(312)
plt.grid(True)
plt.title("Validation Data - Lineal Regression")
plt.xlabel("Income")
plt.ylabel("Predicted Happiness")
plt.scatter(x_test, y_predicted)

plt.show()
