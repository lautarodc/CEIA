import numpy as np

if __name__ == '__main__':
    # Datos de entrada
    n = 100  # Cantidad de simulaciones
    N = 10  # Cantidad de muestras tomadas
    mu = 48  # Media muestral
    sigma = 4  # Desvío

    # Intervalo de confianza teórico
    mu_min = mu - 1.96 * sigma / (N ** 0.5)
    mu_max = mu + 1.96 * sigma / (N ** 0.5)

    trust = 0
    for i in range(n):
        # Generamos valores aleatorios de la distribución
        x = (sigma / (N ** 0.5)) * np.random.randn(N) + mu
        trust = trust + (1 / (n * N)) * np.sum(np.logical_and(x >= mu_min, x <= mu_max), axis=0)

    real_trust = 0.95
    print("Confianza teorica: {t}".format(t=real_trust))
    print("Confianza simulada: {t}".format(t=np.round(trust, 4)))

    n_exp = 100000  # Cantidad de experimentos para el test de hipótesis
    mu_o = 45
    alpha = 0.05
    medias_muestrales = []
    for i in range(n_exp):
        x = np.random.normal(mu_o, sigma, N)
        medias_muestrales.append(np.mean(x, axis=0))
    medias_muestrales = np.array(medias_muestrales)
    prob_media = (1-np.sum(np.round(medias_muestrales) < mu) / n_exp)
    print("Probabilidad simulada de obtener una media muestral X=48: {}".format(prob_media))
    if prob_media < alpha:
        print("Se rechaza la hipótesis nula")
    else:
        print("No se puede rechazar la hipótesis nula")
