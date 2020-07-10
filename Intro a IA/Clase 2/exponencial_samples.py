import numpy as np


def exponencial_n_samples(lambda_param, n_samples):
    n_uniform = np.random.uniform(0, 1, n_samples)
    return (-1 / lambda_param) * np.log(1 - n_uniform)
