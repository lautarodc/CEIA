import numpy as np
import pickle
from sintetic import sintetic_dataset
from add_nan import add_nan
from split_data import split_data
from remove_nan import remove_mean_nan
from exponencial_samples import exponencial_n_samples


class SinteticCluster(object):
    instance = None

    def __new__(cls, dim, size):
        if SinteticCluster.instance is None:
            SinteticCluster.instance = super(SinteticCluster, cls).__new__(cls)
            return SinteticCluster.instance
        else:
            return SinteticCluster.instance

    def __init__(self, dim, size):
        try:
            self.data = pickle.load(open("data_cluster.pickle", "rb"))
            self.clusters = pickle.load(open("clusters.pickle", "rb"))
        except(OSError, IOError) as e:
            # dim centroids for the generation of the synthetic dataset
            # centroids = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            centroids = np.random.randn(4, dim)
            self.centroids = centroids
            data, clusters = sintetic_dataset(centroids, 20, size)
            data = add_nan(data, 0.001)
            train, validation, test, idx_train = split_data(data, 0.7, 0.2)
            pickle.dump(clusters[idx_train], open("clusters.pickle", "wb"))
            pickle.dump(train, open("data_cluster.pickle", "wb"))
            self.data = pickle.load(open("data_cluster.pickle", "rb"))
            self.clusters = pickle.load(open("clusters.pickle", "rb"))

        # pre-processing
        means, self.data = remove_mean_nan(self.data)
        print(self.data.shape[0])
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.norm_2 = np.linalg.norm(self.data, axis=0)
        self.exp_col = exponencial_n_samples(0.4, self.data.shape[0])
        self.exp_col = np.reshape(self.exp_col, (self.data.shape[0], 1))
        self.data = np.append(self.data, self.exp_col, axis=1)

    def get_data(self):
        return self.data

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def get_norm_2(self):
        return self.norm_2

    def get_centroid(self):
        return self.centroids

    def get_exponencial(self):
        return self.exp_col

    def get_clusters(self):
        return self.clusters
