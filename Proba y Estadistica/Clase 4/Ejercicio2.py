import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import binom

n, k = 100, 55
p = np.linspace(0, 1, 1000)
L = np.zeros(p.shape)
DL = np.zeros(p.shape)
dp = p[1] - p[0]

for i, p_binom in enumerate(p):
    L[i] = binom.pmf(55, 100, p_binom)
    if i > 0:
        DL[i] = (L[i] - L[i - 1]) / dp

idx_max = np.argmax(L)
p_max = p[idx_max]
l_max = L[idx_max]

plt.figure(1)
plt.plot(p, L)
plt.title("L=f(p)")
plt.xlabel("p")
plt.ylabel("Likelihood")
plt.annotate('p máxima verosimilitud: %s ' % round(p_max, 2), xy=(p_max, l_max), xytext=(0.2, 0.07),
             arrowprops=dict(facecolor='black', shrink=0.03))

plt.figure(2)
plt.plot(p, DL)
plt.title("dL=f(p)")
plt.xlabel("p")
plt.ylabel("Likelihood derivative")
plt.annotate('p máxima verosimilitud: %s ' % round(p_max, 2), xy=(p_max, DL[idx_max]), xytext=(0.2, 0.07),
             arrowprops=dict(facecolor='black', shrink=0.03))


plt.show()
