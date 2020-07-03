import numpy as np
from norm_IA import norm_vector_l2


def sorting_vector_l2(matrix):
    matrix_l2 = norm_vector_l2(matrix)
    matrix_sorted_asc = matrix[np.argsort(matrix_l2)]
    return matrix_sorted_asc[::-1]
