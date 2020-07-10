import numpy as np
from PCA import pca_numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def test_pca():
    x_test = np.array([[0.4, 4800, 5.5], [0.7, 12104, 5.2], [1, 12500, 5.5], [1.5, 7002, 4.0]])
    x_pca_numpy = pca_numpy(x_test, 2)
    pca = PCA(n_components=2)
    x_std = StandardScaler(with_std=False).fit_transform(x_test)
    x_pca_skl = pca.fit_transform(x_std)
    assert np.testing.assert_allclose(x_pca_numpy, x_pca_skl)
