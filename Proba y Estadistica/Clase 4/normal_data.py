import numpy as np


def generate_data(n, rseed=1, f=0.0, mu=5):
    rand = np.random.RandomState(rseed)
    x_data = rand.randn(n)
    if f > 0:
        x_data[int(f * n):] += mu
    return x_data
