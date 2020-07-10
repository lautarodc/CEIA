import numpy as np
from features_1 import mean_subtraction


def pca_numpy(x, dim):
    x_zscore = mean_subtraction(x)
    x_zscore_t = x_zscore.T
    x_cov = np.cov(x_zscore_t)
    w, v = np.linalg.eig(x_cov)
    asc_w = np.argsort(w)[::-1]
    asc_v = v[:, asc_w]
    asc_w_dim = asc_v[:, :dim]
    return np.matmul(x_zscore, asc_w_dim)
