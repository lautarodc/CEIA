import numpy as np
import matplotlib.pyplot as plt
from DatasetCSV import SplitData
from LinealRegression import LinealRegression
from LinealRegressionAffine import LinealRegressionAffine
from mse_metric import MseMetric

# Data extraction and preprocessing
data_example = SplitData('income.csv')
x_train, y_train = data_example.get_train_data()
x_validation, y_validation = data_example.get_test_data()

# Model fitting
lineal_regression = LinealRegression()
lineal_regression.fit(x_train, y_train)
affine_regression = LinealRegressionAffine()
affine_regression.fit(x_train, y_train)

# Model predictions and metrics
lineal_predictions = lineal_regression.predict(x_validation)
affine_predictions = affine_regression.predict(x_validation)
lineal_results = {"predictions": lineal_predictions, "truth": y_validation}
affine_results = {"predictions": affine_predictions, "truth": y_validation}
lineal_mse = MseMetric(**lineal_results)()
affine_mse = MseMetric(**affine_results)()
lineal_mse = np.round(lineal_mse, 4)
affine_mse = np.round(affine_mse, 4)

# Graphics

font = {
    'family': 'DejaVu sans',
    'color': 'black',
    'weight': 'normal',
    'size': 9
}

plt.figure(1)
plt.scatter(data_example.get_data()['income'], data_example.get_data()['happiness'])
plt.grid(True)
plt.title("Happiness as f(income) (all data) ")
plt.xlabel("Income")
plt.ylabel("Happiness")

plt.figure(2)

plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title("Validation Data")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.scatter(x_validation, y_validation)

plt.subplot(312)
plt.grid(True)
plt.title("Validation Data - Lineal Regression")
plt.xlabel("Income")
plt.ylabel("Predicted Happiness")
plt.text(2, 4, 'Mean Square Error = %s' % lineal_mse, fontdict=font)
plt.scatter(x_validation, lineal_predictions)

plt.subplot(313)
plt.grid(True)
plt.title("Validation Data - Affine Regression")
plt.xlabel("Income")
plt.ylabel("Predicted Happiness")
plt.text(2, 4, 'Mean Square Error = %s' % affine_mse, fontdict=font)
plt.scatter(x_validation, affine_predictions)

plt.show()
