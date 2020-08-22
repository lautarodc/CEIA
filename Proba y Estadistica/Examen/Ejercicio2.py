import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


def inverse_sampling(n_samples):
    n_uniform = np.random.uniform(0, 1, n_samples)
    return 3 * np.sqrt(n_uniform)


if __name__ == '__main__':
    sns.set()
    # Generar los datos para nuestra distribución mediante el método de la transformada inversa
    N = 1000
    x = inverse_sampling(N)

    # Estimación de densidad de kernel - Gaussiana Manual
    h = 1  # Coeficiente de dispersión de la gaussiana, se coloca aquí en 1 para facilitar el cálculo

    # Generamos el vector xd para las Gaussianas
    xd = np.linspace(-0.1, 3.1, 1000)
    # Obtenemos la pdf como la suma de los kernels centrados en cada punto y normalizados
    density = sum(norm(xi).pdf(xd) for xi in x) * (1 / x.shape[0])
    # Graphic
    plt.figure(1)
    plt.fill_between(xd, density, alpha=0.5)
    plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
    plt.title("KDE Manual")
    # Cálculo del MSE
    pdf = (2/9)*xd
    mse_manual = np.sum((pdf - density) ** 2, axis=0) * (1 / density.shape[0])
    print("MSE Manual, h=1: {mse}".format(mse=mse_manual))

    # Scikit-Learn Kernel Density Estimation

    # Instanciamos el modelo y hacemos el fit
    kde = KernelDensity(bandwidth=0.201, kernel='gaussian')
    kde.fit(x[:, None])

    # Obtenemos el logarítmo de las probabilidades evaluado en xd
    logprob = kde.score_samples(xd[:, None])

    # Gráfica
    plt.figure(2)
    plt.fill_between(xd, np.exp(logprob), alpha=0.5)
    plt.title("KDE de Scikit-Learn")

    # Cálculo del MSE
    mse_skl = np.sum((pdf - np.exp(logprob)) ** 2, axis=0) * (1 / pdf.shape[0])
    print("MSE Manual, h=0.2: {mse}".format(mse=mse_skl))

    # MSE en función del ancho de banda
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    mse_bandwidths = np.zeros(bandwidths.shape)
    for i, bandwidth in enumerate(bandwidths):
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(x[:, None])
        logprob = kde.score_samples(xd[:, None])
        mse_bandwidths[i] = np.sum((pdf - np.exp(logprob)) ** 2, axis=0) * (1 / pdf.shape[0])

    plt.figure(3)
    plt.plot(bandwidths, mse_bandwidths)
    plt.title("MSE vs. KDE para diferentes anchos de banda")
    plt.xlabel("Ancho de banda")
    plt.ylabel("MSE")

    # Ancho de banda con menor MSE KDE
    min_idx = np.argmin(mse_bandwidths)
    min_mse = np.min(mse_bandwidths)
    best_n_bin = bandwidths[min_idx]
    print("Ancho de banda mínimo: ")
    print(best_n_bin)
    print("MSE mínimo")
    print(min_mse)

    plt.show()
