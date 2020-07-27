import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from normal_data import generate_data

sns.set()


# Generate normal distributed data of mu=0 and std=1
x = generate_data(1000)

# Generate a vector of number of bins
n_bins = np.linspace(1, 50, 10)

# Graphics
font = {
    'family': 'DejaVu sans',
    'color': 'black',
    'weight': 'normal',
    'size': 9
}

# Histogram Graphics for different N° of Bins
fig, ax = plt.subplots(1, n_bins.shape[0], sharey='all', sharex='all')
fig.subplots_adjust(wspace=0.05)

st = fig.suptitle("Histograms of x for different n° of bins", fontsize="x-large")

for i, n_bin in enumerate(n_bins):
    ax[i].text(-2, 0.5, 'N° Bins = %s' % int(n_bin), fontdict=font)
    ax[i].hist(x, bins=int(n_bin), density=True)
    ax[i].plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

st.set_y(0.95)
fig.subplots_adjust(top=0.85)

# MSE of Histogram Density Estimator as a function of N° bins
n_bins_mse = np.linspace(1, 1000, 500)
bin_width = np.zeros(n_bins_mse.shape)
mse_bins = np.zeros(n_bins_mse.shape)

for i, n_bin in enumerate(n_bins_mse):
    density, bins = np.histogram(x, int(n_bin), density=True)
    bin_width[i] = bins[1] - bins[0]
    x_bins = bins[0:-1] + bin_width[i]
    pdf = norm.pdf(x_bins)
    mse_bins[i] = np.sum((pdf - density) ** 2, axis=0) * (1 / x_bins.shape[0])

plt.figure(2)
plt.plot(n_bins_mse, mse_bins)
plt.title("MSE vs. Histograms of x for different n° of bins")
plt.xlabel("N° of bins")
plt.ylabel("MSE")

min_idx = np.argmin(mse_bins)
min_mse = np.min(mse_bins)
best_n_bin = n_bins_mse[min_idx]
density, bins = np.histogram(x, int(best_n_bin), density=True)
bin_width = bins[1] - bins[0]
x_bins = bins[0:-1] + bin_width
print(best_n_bin)
print(min_mse)
plt.figure(3)
xd = np.linspace(-5, 5, 1000)
plt.fill_between(x_bins, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)


plt.show()
