import numpy as np


def add_nan(x, p):
    n_elements = int(np.size(x) * p)
    x_nan = np.array(x, copy=True)
    x_nan.ravel()[np.random.choice(x_nan.size, n_elements, replace=False)] = np.NaN
    return x_nan
