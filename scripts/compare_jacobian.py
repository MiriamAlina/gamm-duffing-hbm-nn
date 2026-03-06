import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.nn_jacobian import NN_jacobian_Duffing_H3


# different ordering conventions for the Jacobian rows (output variables):
# NN order: (a0, a1, b1, a2, b2, a3, b3)
# AFT order: (a0, a1, a2, a3, b1, b2, b3)
perm_row_0 = np.array([0, 1, 3, 5, 2, 4, 6])
alpha = 0.5  # scaling factor due to conversions from ce to cs within AFT

inv_perm = np.empty_like(perm_row_0)
inv_perm[perm_row_0] = np.arange(len(perm_row_0))


###############################################################################
# Load FD Jacobian saved from MATLAB
###############################################################################
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
NN_id = '2026-02-18_13-29-30'
evaluate_coefficients = False

J_nn = np.empty((K, 7, 7))
for k in range(K):
    Om = float(X_full[k, -1])
    J_nn[k] = NN_jacobian_Duffing_H3(X[k, :], mu, zeta, kappa, gamma, P, H, N,
                                     'NN', NN_id, evaluate_coefficients, Om)


###############################################################################
# Plot relevant gradients: A1,B1,A3,B3 w.r.t a1,b1,a3,b3
###############################################################################
def zero_clean_formatter(x, pos):
    if abs(x) < 1e-12:
        return "0"
    return f"{x:g}"


idx = [1, 2, 5, 6]  # (a1,b1,a3,b3) in NN-order
input_labels = [r"$a_1$", r"$b_1$", r"$a_3$", r"$b_3$"]
input_symbols = [r"a_1", r"b_1", r"a_3", r"b_3"]
output_symbols = [r"A_1", r"B_1", r"A_3", r"B_3"]

fig, axes = plt.subplots(4, 4, figsize=(10, 7), sharex="col")
for ii, i in enumerate(idx):  # output index (row)
    for jj, j in enumerate(idx):  # input index (col)
        ax = axes[ii, jj]
        x = X[:, j]
        y_fd = Jsub_fd_nnorder[:, i, j]
        y_nn = J_nn[:, i, j]
        ax.scatter(x, y_fd, s=8, alpha=0.4, color="#A8DADC",
                   label="Finite Differences" if (ii == 0 and jj == 0)
                   else None)
        ax.scatter(x, y_nn, s=8, alpha=0.4, color="#E63946",
                   label="Neural Network" if (ii == 0 and jj == 0)
                   else None)
        if ii == 3:
            ax.set_xlabel(input_labels[jj], fontsize=12)

        ylabel = (fr"$\frac{{\partial {output_symbols[ii]}}}"
                  fr"{{\partial {input_symbols[jj]}}}$")
        ax.set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=15,
                      va="center")

for ax in axes.flat:
    ax.xaxis.set_major_formatter(FuncFormatter(zero_clean_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(zero_clean_formatter))

handles, labels_legend = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_legend, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)
fig.subplots_adjust(wspace=0.8, hspace=0.45)
fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figures/jacobian_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


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

print("FD: min smin =", np.min(smin_fd), " max cond =", np.max(cond_fd))
print("NN: min smin =", np.min(smin_nn), " max cond =", np.max(cond_nn))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].semilogy(smin_fd, '.', label='Finite Differences', alpha=0.7,
                 color="#A8DADC")
axes[0].semilogy(smin_nn, '.', label='Neural Network', alpha=0.7,
                 color="#E63946")
axes[0].set_title("Smallest singular value")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel(r"$\sigma_{\min}(J)$")
axes[0].grid(True)

axes[1].semilogy(cond_fd, '.', label='Finite Differences', alpha=0.7,
                 color="#A8DADC")
axes[1].semilogy(cond_nn, '.', label='Neural Network', alpha=0.7,
                 color="#E63946")
axes[1].set_title("Condition number")
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel(r"$\kappa(J)$")
axes[1].grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figures/jacobian_conditioning.svg", bbox_inches="tight")
plt.show()


###############################################################################
# Conditioning vs Omega
###############################################################################
Omega = X_full[:, -1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].semilogy(Omega, smin_fd, '.', label='Finite Differences', alpha=0.7,
                 color="#A8DADC")
axes[0].semilogy(Omega, smin_nn, '.', label='Neural Network', alpha=0.7,
                 color="#E63946")
axes[0].set_title("Smallest singular value vs Ω")
axes[0].set_xlabel(r"$\Omega$")
axes[0].set_ylabel(r"$\sigma_{\min}(J)$")
axes[0].grid(True)

axes[1].semilogy(Omega, cond_fd, '.', label='Finite Differences', alpha=0.7,
                 color="#A8DADC")
axes[1].semilogy(Omega, cond_nn, '.', label='Neural Network', alpha=0.7,
                 color="#E63946")
axes[1].set_title("Condition number vs Ω")
axes[1].set_xlabel(r"$\Omega$")
axes[1].set_ylabel(r"$\kappa(J)$")
axes[1].grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figures/jacobian_conditioning_vs_omega.svg", bbox_inches="tight")
plt.show()


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

print("Linearisation quality")
print("FD mean:", np.mean(lin_err_fd))
print("NN mean:", np.mean(lin_err_nn))


###############################################################################
# Jacobian smoothness
###############################################################################
smooth_fd = []
smooth_nn = []

for k in range(len(J_nn)-1):

    smooth_fd.append(
        np.linalg.norm(Jsub_fd_nnorder[k+1] - Jsub_fd_nnorder[k])
    )

    smooth_nn.append(
        np.linalg.norm(J_nn[k+1] - J_nn[k])
    )

print("\nJacobian smoothness")
print("FD mean change:", np.mean(smooth_fd))
print("NN mean change:", np.mean(smooth_nn))
