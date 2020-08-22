import numpy as np


def experiment():
    # Se generan las muestras con una probabilidad de falla p de 0.01
    n_samples = 10000
    samples = np.zeros((n_samples, 1))
    p = 0.01
    samples[:int(n_samples * p), :] = 1
    permute_ids = np.random.permutation(samples.shape[0])  # Se hace un shuffle de las muestras
    samples = samples[permute_ids]

    # Generamos un vector de N cantidad de fósforos
    n_vector = np.linspace(1, 100, 100)
    EPOCH = 1000  # Cantidad de experimentos
    p_vector = np.zeros_like(n_vector)  # Vector para almacenar probabilidad de obtener uno o más fósforos defectuosos
    fail_cases = np.zeros((EPOCH, 1))  # Vector para almacenar la cantidad de fósforos defectuosos por EPOCH

    for i, n in enumerate(n_vector):
        for j in range(EPOCH):
            idx = np.random.randint(0, n_samples, int(n))  # Obtenemos N índices aleatorios del vector de muestras
            fail_cases[j] = np.sum(samples[idx] == 1, axis=0)  # Cantidad de fósforos defectuosos
        p_vector[i] = np.sum(fail_cases > 0, axis=0) / fail_cases.shape[0] # Proporción de fósforos defectuosos
        if p_vector[i] >= 0.5:
            n_ok = i
            break
    return n_ok


if __name__ == '__main__':
    # Realizamos varios experimentos para reducir el efecto de la aleatoreidad
    TRIALS = 1
    n_trials = []
    for i in range(TRIALS):
        n_trials.append(experiment())
    n = np.array(n_trials)
    n_unique = np.unique(n)
    n_count = [n_trials.count(x) for x in n_unique]
    result = np.array((n_unique, n_count))
    print(result)

    # Generamos N muestras bernoulli con p=0.01
    EPOCHS = 1000
    N = 68
    p = 0.01
    mean_epoch = []
    stdv_epoch = []
    for i in range(EPOCHS):
        samples = np.random.binomial(N, p, 1000)
        mean_epoch.append(np.mean(samples, axis=0))
        stdv_epoch.append(np.std(samples, axis=0))
    mean_epoch = np.array(mean_epoch)
    stdv_epoch = np.array(stdv_epoch)
    print("Media simulada:{media} Desvío simulado: {std}".format(media=np.round(np.mean(mean_epoch), 4),
                                                                 std=np.round(np.mean(stdv_epoch), 4)))
    print("Media teórica:{media} Desvío teórico: {std}".format(media=N * p, std=np.round((N * p * (1 - p)) ** 0.5, 4)))
