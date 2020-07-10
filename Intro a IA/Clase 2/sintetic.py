import numpy as np


def sintetic_dataset(centroides, desvio, num_muestras):
    centroides_desv = centroides * desvio
    muestras = np.repeat(centroides_desv, num_muestras / 4, axis=0)
    normal_noise = np.random.normal(0, 1, size=muestras.shape)
    muestras_noise = muestras + normal_noise
    cluster_ids = np.array([[0], [1], [2], [3]])
    cluster_ids = np.repeat(cluster_ids, num_muestras/4, axis=0)
    return muestras_noise, cluster_ids
