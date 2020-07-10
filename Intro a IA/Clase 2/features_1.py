import numpy as np


def z_score_dataset(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def mean_subtraction(x):
    mean = np.mean(x, axis=0)
    return x - mean
