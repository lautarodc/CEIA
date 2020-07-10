import numpy as np


def distance_center(x, c):
    c_expanded = c[:, None]
    return np.round(np.sqrt(np.sum((c_expanded - x) ** 2, axis=2)), 4)


def minimum_distance_center(x: np.ndarray) -> np.ndarray:
    return np.argmin(x, axis=0)
