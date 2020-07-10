import numpy as np


def cubic_variable_generator(n_samples):
    n_uniform = np.random.uniform(0, 1, n_samples)
    return np.cbrt(n_uniform)
