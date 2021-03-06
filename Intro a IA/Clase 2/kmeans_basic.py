import numpy as np
from distance_center import distance_center, minimum_distance_center


def k_means(x_data, n_clusters, max_iterations):
    np.random.seed(1)
    n_indexes = np.random.randint(0, x_data.shape[0], n_clusters)
    n_centers = x_data[n_indexes]
    for i in range(max_iterations):
        x_distances = distance_center(x_data, n_centers)
        x_cluster_id = minimum_distance_center(x_distances)
        for j in range(n_centers.shape[0]):
            n_centers[j] = np.mean(x_data[x_cluster_id == j, :], axis=0)
    return n_centers, x_cluster_id
