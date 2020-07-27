import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from normal_data import generate_data
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut

sns.set()

# Generate normal distributed data of mu=0 and std=1
x = generate_data(100, f=0, mu=0)

# Manual Kernel Density Estimation

# Generate a xd vector for the gaussian kernel
xd = np.linspace(-3, 3, 1000)
# Compute the density as the sum of the Gaussian Kernels centered at each point and normalized
density = sum(norm(xi).pdf(xd) for xi in x) * (1 / x.shape[0])
# Graphic
plt.figure(1)
plt.fill_between(xd, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
plt.title("KDE Manual")
# MSE calculation
pdf = norm.pdf(xd)
mse_manual = np.sum((pdf - density) ** 2, axis=0) * (1 / density.shape[0])
print(mse_manual)

# Scikit-Learn Kernel Density Estimation

# Instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])

# Return the log of the probability density evaluated in xd
logprob = kde.score_samples(xd[:, None])

# Graphic
plt.figure(2)
plt.fill_between(xd, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.title("KDE de Scikit-Learn")

# MSE Calculation
mse_skl = np.sum((pdf - np.exp(logprob)) ** 2, axis=0) * (1 / pdf.shape[0])
print(mse_skl)

# Evaluate the MSE as a function of bandwidth
bandwidths = 10 ** np.linspace(-1, 1, 100)
mse_bandwidths = np.zeros(bandwidths.shape)
for i, bandwidth in enumerate(bandwidths):
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(x[:, None])
    logprob = kde.score_samples(xd[:, None])
    mse_bandwidths[i] = np.sum((pdf - np.exp(logprob)) ** 2, axis=0) * (1 / pdf.shape[0])

plt.figure(3)
plt.plot(bandwidths, mse_bandwidths)
plt.title("MSE vs. KDE for different bandwidths")
plt.xlabel("Bandwidth")
plt.ylabel("MSE")

# Bandwidth with minimum MSE of KDE
min_idx = np.argmin(mse_bandwidths)
min_mse = np.min(mse_bandwidths)
best_n_bin = bandwidths[min_idx]
print("Minimum Bandwidth: ")
print(best_n_bin)
print(min_mse)

# Best Bandwidth Calculation - Scikit-Learn Cross Validation and Grid Search
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut())
grid.fit(x[:, None])
print(grid.best_params_)

kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
kde.fit(x[:, None])
logprob_best = kde.score_samples(xd[:, None])
mse_best_scikit = np.sum((pdf - np.exp(logprob_best)) ** 2, axis=0) * (1 / pdf.shape[0])
print(mse_best_scikit)

plt.show()
