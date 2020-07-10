import numpy as np
from add_nan import add_nan


def remove_row_nan(x):
    x_nan = x[~np.isnan(np.sum(x, axis=1))]
    return x_nan


def remove_mean_nan(x):
    mean_features = np.reshape(np.nanmean(x, axis=0), (1, 4))
    mean_matrix = np.repeat(mean_features, x.shape[0], axis=0)
    x_nan = np.array(x, copy=True)
    x_nan[np.isnan(x)] = mean_matrix[np.isnan(x)]
    return mean_features, x_nan
