import numpy as np
import matplotlib.pyplot as plt
from src.nn_jacobian import NN_jacobian_Duffing_H3

# ============================================================
# Goal:
# - Keep the NN Jacobian in its original, intuitive order
#   (a0, a1, b1, a2, b2, a3, b3) so that indices [1,2,5,6] mean
#   (a1, b1, a3, b3) as you expect.
# - Therefore: "unshuffle" the AFT/FD Jacobian rows to match NN order.
# - Then plot ONLY the 4x4 nonzero block for (a1,b1,a3,b3) -> (a1,b1,a3,b3)
# ============================================================

# --- NN/AFT alignment info you found in MATLAB ---------------------------
# MATLAB: perm_row = [1 2 4 6 3 5 7], alpha = 0.5 meant:
#   JA (AFT) ≈ alpha * JB(perm_row,:)
# So to bring AFT FD rows into NN ordering:
#   J_FD_NNorder = (1/alpha) * J_FD_AFTorder[inv_perm_row, :]
# 0-based version of [1 2 4 6 3 5 7]
perm_row_0 = np.array([0, 1, 3, 5, 2, 4, 6])
alpha = 0.5

inv_perm = np.empty_like(perm_row_0)
inv_perm[perm_row_0] = np.arange(len(perm_row_0))

# --- load FD Jacobians saved from MATLAB ---------------------------------
m, n = 7, 8
Jsize = m * n

jac_FD = np.loadtxt('data/jac_FD_Duffing.txt', delimiter=',')
J_flat = jac_FD[:, :Jsize]         # (K, 56)
X_full = jac_FD[:, Jsize:]         # (K, 8)  [a0,a1,b1,a2,b2,a3,b3,Omega]
K = jac_FD.shape[0]

J_all = np.empty((K, m, n))
for k in range(K):
    J_all[k] = J_flat[k].reshape((m, n), order="F")  # MATLAB column-major

X = X_full[:, :7]
Jsub_fd = J_all[:, :7, :7]  # drop Omega-col => 7x7

# --- convert FD Jacobian from AFT-order to NN-order ----------------------
# (only rows need reordering; columns correspond to variables already in
# [a0,a1,b1,a2,b2,a3,b3])
Jsub_fd_nnorder = (1.0 / alpha) * Jsub_fd[:, inv_perm, :]

# --- NN params -----------------------------------------------------------
mu = 1
kappa = 1
zeta = 0.05
gamma = 0.1
P = 0.18
H = 3
N = 4 * H + 1
NN_id = '2026-02-18_13-29-30'
evaluate_coefficients = False

# --- compute NN Jacobians (NO shuffling/scaling here) --------------------
J_nn = np.empty((K, 7, 7))
for k in range(K):
    Om = float(X_full[k, -1])
    J_nn[k] = NN_jacobian_Duffing_H3(
        X[k, :],
        mu, zeta, kappa, gamma, P, H, N,
        'NN', NN_id, evaluate_coefficients, Om
    )

# --- optional sanity number (should be ~0.08-0.10) -----------------------
rel_fro = np.linalg.norm(Jsub_fd_nnorder - J_nn) / \
    np.linalg.norm(Jsub_fd_nnorder)
print("global rel fro error (FD rows put into NN-order vs NN):", rel_fro)

# --- plot ONLY the relevant 4x4 block: indices 1,2,5,6 => a1,b1,a3,b3 ----
idx = [1, 2, 5, 6]  # (a1,b1,a3,b3) in NN-order
input_labels = [r"$a_1$", r"$b_1$", r"$a_3$", r"$b_3$"]
output_labels = [r"$\partial A_1$", r"$\partial B_1$", r"$\partial A_3$",
                 r"$\partial B_3$"]

fig, axes = plt.subplots(4, 4, figsize=(12, 8), sharex="col")

for ii, i in enumerate(idx):          # output index (row)
    for jj, j in enumerate(idx):      # input index  (col)
        ax = axes[ii, jj]
        x = X[:, j]
        y_fd = Jsub_fd_nnorder[:, i, j]
        y_nn = J_nn[:, i, j]
        ax.scatter(x, y_fd, s=8, alpha=0.4, color="#A8DADC",
                   label="Finite Differences" if (ii == 0 and jj == 0) else
                   None)
        ax.scatter(x, y_nn, s=8, alpha=0.4, color="#E63946",
                   label="Neural Network" if (ii == 0 and jj == 0) else None)
        if ii == 3:
            ax.set_xlabel(input_labels[jj])
        ax.set_ylabel(output_labels[ii]+"\n---\n"+r"$\partial$" +
                      input_labels[jj], rotation=0, labelpad=15)


handles, labels_legend = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_legend, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(wspace=0.8, hspace=0.2)
plt.tight_layout()
plt.savefig("figures/jacobian_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
