import numpy as np
import matplotlib.pyplot as plt
from src.nn_jacobian import NN_jacobian_Duffing_H3


path = "jac_FD_Duffing.txt"
m, n = 7, 8
Jsize = m * n
Xsize = n
jac_FD = np.loadtxt('data/jac_FD_Duffing.txt', delimiter=',')
J_flat = jac_FD[:, :Jsize]     # (K, 56)
X_full = jac_FD[:, Jsize:]     # (K, 8)
K = jac_FD.shape[0]

J_all = np.empty((K, m, n))
for k in range(K):
    J_all[k] = J_flat[k].reshape((m, n), order="F")  # column-wise order='F'

# X_full = [x0, x1, x2, x3, x4, x5, x6, Omega]
X = X_full[:, :7]
Jsub = J_all[:, :7, :7]

mu = 1
kappa = 1
zeta = 0.05
gamma = 0.1
P = 0.18
H = 3
N = 4*H+1
NN_id = '2026-02-18_13-29-30'
evaluate_coefficients = False

J_nn_7 = np.empty((K, 7, 7))

for k in range(K):
    Om = float(X_full[k, -1])
    J_nn_7[k] = NN_jacobian_Duffing_H3(
        X[k, :],
        mu, zeta, kappa, gamma, P, H, N,
        'NN', NN_id, evaluate_coefficients, Om
    )

fig, axes = plt.subplots(7, 7, figsize=(14, 8), sharex="col")

for i in range(7):
    for j in range(7):
        ax = axes[i, j]
        x = X[:, j]                 # x_j
        y_fd = Jsub[:, i, j]      # FD: J_ij
        y_nn = J_nn_7[:, i, j]      # NN: J_ij

        ax.scatter(x, y_fd, s=6, alpha=0.35, color='#A8DADC',
                   label="FD" if (i == 0 and j == 0) else None)
        ax.scatter(x, y_nn, s=6, alpha=0.35, color='#E63946',
                   label="NN" if (i == 0 and j == 0) else None)

        if i == 6:
            ax.set_xlabel(f"x{j+1}")
        if j == 0:
            ax.set_ylabel(f"J[{i+1},{j+1}]")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.suptitle("FD vs NN: Jacobian entry J[i,j] plotted against input x_j")
plt.tight_layout()
plt.show()
