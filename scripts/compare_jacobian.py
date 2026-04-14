import numpy as np
import matplotlib.pyplot as plt
from src.util import check_folder_structure
from src.nn_jacobian import NN_jacobian_Duffing_H3
from src.plotting import (gradients_over_inputs_plot,
                          smin_over_omega,
                          cond_over_omega)


###############################################################################
# Check folder structure
###############################################################################
check_folder_structure()


###############################################################################
# Load FD Jacobian saved from MATLAB
###############################################################################
# different ordering conventions for the Jacobian rows (output variables):
# NN order: (a0, a1, b1, a2, b2, a3, b3)
# AFT order: (a0, a1, a2, a3, b1, b2, b3)
perm_row_0 = np.array([0, 1, 3, 5, 2, 4, 6])
alpha = 0.5  # scaling factor due to conversions from ce to cs within AFT

inv_perm = np.empty_like(perm_row_0)
inv_perm[perm_row_0] = np.arange(len(perm_row_0))

m, n = 7, 8
Jsize = m * n

jac_FD = np.loadtxt('data/jacobian_input_frc_aft.txt', delimiter=',')
J_flat = jac_FD[:, :Jsize]         # (K, 56)
X_full = jac_FD[:, Jsize:]         # (K, 8)  [a0,a1,b1,a2,b2,a3,b3,Omega]
K = jac_FD.shape[0]

J_all = np.empty((K, m, n))
for k in range(K):
    J_all[k] = J_flat[k].reshape((m, n), order="F")  # MATLAB column-major

X = X_full[:, :7]
Jsub_fd = J_all[:, :7, :7]  # drop Omega-col => 7x7

# convert FD Jacobian from AFT-order to NN-order
# (only rows need reordering; columns correspond to variables already in
# [a0,a1,b1,a2,b2,a3,b3])
Jsub_fd_nnorder = (1.0 / alpha) * Jsub_fd[:, inv_perm, :]


###############################################################################
# Compute NN Jacobian
###############################################################################
mu = 1
kappa = 1
zeta = 0.05
gamma = 0.1
P = 0.18
H = 3
N = 4 * H + 1
NN_id = '2026-04-14_12-00-35'
evaluate_coefficients = False

J_nn = np.empty((K, 7, 7))
for k in range(K):
    Om = float(X_full[k, -1])
    J_nn[k] = NN_jacobian_Duffing_H3(X[k, :], mu, zeta, kappa, gamma, P, H, N,
                                     'NN', NN_id, evaluate_coefficients, Om)


###############################################################################
# Plot relevant gradients: A1,B1,A3,B3 w.r.t a1,b1,a3,b3
###############################################################################
gradients_over_inputs_plot(X, Jsub_fd_nnorder, J_nn,
                           figure_name="jacobian_comparison",
                           save_figure=False)


###############################################################################
# Compare conditioning: smallest singular value and condition number
###############################################################################
smin_fd = np.empty(K)
smin_nn = np.empty(K)
cond_fd = np.empty(K)
cond_nn = np.empty(K)

for k in range(K):
    sv_fd = np.linalg.svd(Jsub_fd_nnorder[k], compute_uv=False)
    sv_nn = np.linalg.svd(J_nn[k], compute_uv=False)
    smin_fd[k] = sv_fd[-1]
    smin_nn[k] = sv_nn[-1]
    cond_fd[k] = sv_fd[0] / max(sv_fd[-1], 1e-14)
    cond_nn[k] = sv_nn[0] / max(sv_nn[-1], 1e-14)

print("Smallest singular values")
print("FD: min smin =", np.min(smin_fd))
print("NN: min smin =", np.min(smin_nn))

print("\nCondition numbers")
print("FD: max cond =", np.max(cond_fd))
print("NN: max cond =", np.max(cond_nn))


###############################################################################
# Conditioning vs Omega
###############################################################################
Omega = X_full[:, -1]

smin_over_omega([smin_fd, smin_nn], Omega, np.min([smin_fd, smin_nn]),
                np.max([smin_fd, smin_nn]),
                figure_name="jacobian_smin_vs_omega", save_figure=False)
cond_over_omega([cond_fd], Omega, np.min([cond_fd, cond_nn]),
                np.max([cond_fd, cond_nn]),
                figure_name="jacobian_fd_cond_vs_omega", save_figure=False)
cond_over_omega([cond_fd, cond_nn], Omega, np.min([cond_fd, cond_nn]),
                np.max([cond_fd, cond_nn]),
                figure_name="jacobian_fd_nn_cond_vs_omega", save_figure=False)


###############################################################################
# Linearisation quality test
###############################################################################
def linearisation_error(J, R, dx):
    return np.linalg.norm(R + J @ dx) / max(np.linalg.norm(R), 1e-14)


lin_err_fd = []
lin_err_nn = []

for k in range(len(X)):
    Jfd = Jsub_fd_nnorder[k]
    Jnn = J_nn[k]
    dx = np.random.randn(7)
    dx /= np.linalg.norm(dx)
    dx *= 1e-6
    Rfd = Jfd @ dx
    Rnn = Jnn @ dx
    lin_err_fd.append(np.linalg.norm(Rfd))
    lin_err_nn.append(np.linalg.norm(Rnn))

print("\nLinearisation quality")
print("FD mean:", np.mean(lin_err_fd))
print("NN mean:", np.mean(lin_err_nn))


###############################################################################
# Jacobian smoothness
###############################################################################
smooth_fd = []
smooth_nn = []

for k in range(len(J_nn)-1):
    smooth_fd.append(np.linalg.norm(Jsub_fd_nnorder[k+1] - Jsub_fd_nnorder[k]))
    smooth_nn.append(np.linalg.norm(J_nn[k+1] - J_nn[k]))

print("\nJacobian smoothness")
print("FD mean change:", np.mean(smooth_fd))
print("NN mean change:", np.mean(smooth_nn))

plt.show()
