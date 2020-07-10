from clustered_dataset import SinteticCluster
from PCA import pca_numpy
from kmeans_basic import k_means
import matplotlib.pyplot as plt

# Creates the synthetic dataset
clustered_dataset = SinteticCluster(4, 100000)
data = clustered_dataset.get_data()
clusters_train = clustered_dataset.get_clusters()
exp_data = clustered_dataset.get_exponencial()

# Histogram of the exponential feature of the dataset
plt.figure(1)
n, bins, patches = plt.hist(exp_data, 100, density=1, color='g')
plt.title('Histogram of Exponential Distribution Column in Clustered Dataset')
plt.grid(True)

# Apply PCA to reduce the dataset in 2 dimensions
plt.figure(2)
plt.subplot(311)
pca_data = pca_numpy(data, 2)
plt.scatter(pca_data[:, 0], pca_data[:, 1], clusters_train)
plt.title('Scatter plot of the PCA clustered data')
plt.grid(True)

# Apply K-means to original data
centers, clusters = k_means(data, 4, 10)
print(clusters.shape[0])
plt.subplot(312)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
plt.grid(True)
plt.title('Scatter plot of k-means clusters of the original data')

# Apply K-means to pca data
centers_pca, clusters_pca = k_means(pca_data, 4, 10)
print(clusters_pca.shape[0])
plt.subplot(313)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters_pca)
plt.grid(True)
plt.title('Scatter plot of k-means clusters of the pca data')

# TODO: include in the plot the coordinates of each centroid

plt.show()
