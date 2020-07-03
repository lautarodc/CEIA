import numpy as np


def norm_vector_l0(matrix):
    mask = matrix != 0
    return np.sum(mask, axis=1)


def norm_vector_l1(matrix):
    return np.sum(np.abs(matrix), axis=1)


def norm_vector_l2(matrix):
    return np.sqrt(np.sum(matrix * matrix, axis=1))


def norm_vector_inf(matrix):
    return np.max(matrix, axis=1)
