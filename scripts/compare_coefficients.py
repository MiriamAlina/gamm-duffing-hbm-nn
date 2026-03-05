import numpy as np
import matplotlib.pyplot as plt
from src.aft import compute_AFT_solution
from src.fourier_conversion import (convert_cossin_to_comexp,
                                    convert_comexp_to_cossin)
from src.nn_inference import evaluate_Duffing_nn_H3
from src.error_metrics import compute_error_metrics


###############################################################################
# Performance on test data
###############################################################################
test_data_id = '2026-02-18_14-04-47'
test_data = np.load('data/duffing_test_data_H3_N64_' + test_data_id + '.npz')
q_test = test_data['q_coeffs']
fnl_test_aft = test_data['fnl_coeffs']

nn_id = '2026-02-18_13-29-30'
fnl_test_nn = []
for i in range(len(q_test)):
    fnl_test_nn.append(evaluate_Duffing_nn_H3(nn_id, q_test[i, :]))
fnl_test_nn = np.array(fnl_test_nn)

global_metrics_test, individual_metrics_test = \
    compute_error_metrics(fnl_test_aft, fnl_test_nn)
global_metrics_test_normalized, individual_metrics_test_normalized = \
    compute_error_metrics(fnl_test_aft, fnl_test_nn, normalize=True)

# Spider plot of error metrics of test samples
labels = list(global_metrics_test.keys())
values = [global_metrics_test[k] for k in labels]
values_norm = [global_metrics_test_normalized[k] for k in labels]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
values = np.r_[values, values[0]]
values_norm = np.r_[values_norm, values_norm[0]]
angles = np.r_[angles, angles[0]]
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
ax.set_yscale("log")
ax.plot(angles, values, marker="o", label="raw", color='#1D3557')
ax.fill(angles, values, alpha=0.2, color='#1D3557')
ax.plot(angles, values_norm, marker="o", label="normalized", color='#E63946')
ax.fill(angles, values_norm, alpha=0.2, color='#E63946')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.tick_params(axis='x', pad=12)
ax.set_ylim(0, max(values + values_norm)*1.1)
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)
plt.tight_layout()
plt.savefig('figures/error_metrics_spider_test.svg', bbox_inches='tight')
plt.show()

# bar plot of individual error metrics per output of test samples
outputs = ["A1", "B1", "A3", "B3"]
colors = ['#1D3557', '#008b9a', '#f19699', '#e63946']
metrics = individual_metrics_test.keys()
raw = np.vstack([individual_metrics_test[k] for k in metrics])
norm = np.vstack([individual_metrics_test_normalized[k] for k in metrics])
x = np.arange(len(metrics))
w = 0.18
fig, ax = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
for k, (title, data) in enumerate(zip(["raw", "normalized"], [raw, norm])):
    for j in range(len(outputs)):
        ax[k].bar(x + (j-1.5)*w, data[:, j], width=w, label=outputs[j],
                  color=colors[j])
    ax[k].set_xticks(x)
    ax[k].set_xticklabels(metrics)
    ax[k].set_title(title)
ax[0].set_ylabel("error (0 = better)")
ax[1].legend(title="output")
plt.tight_layout()
plt.savefig('figures/error_metrics_bar_test.svg', bbox_inches='tight')
plt.show()


###############################################################################
# Performance on FRC trajectory
###############################################################################
q_frc_full = np.loadtxt('data/nn_input_Duffing.txt', delimiter=',')
q_rel = np.concatenate([q_frc_full[:, 1:3], q_frc_full[:, 5:7]], axis=1)

H = 3
N = 2**6
gamma = 0.1
fnl_rel_aft = np.empty((0, 4))
fnl_rel_nn = np.empty((0, 4))
for i in range(np.shape(q_frc_full)[0]):
    q_ce = convert_cossin_to_comexp(q_frc_full[i, :7])
    fnl_ce = compute_AFT_solution(N, H, q_ce, gamma)
    fnl_cs = convert_comexp_to_cossin(fnl_ce, H)
    fnl_rel_aft = np.vstack([fnl_rel_aft, fnl_cs[[1, 2, 5, 6]]])

    fnl_cs_NN = evaluate_Duffing_nn_H3(nn_id, q_rel[i])
    fnl_rel_nn = np.vstack([fnl_rel_nn, fnl_cs_NN])

global_metrics_frc, individual_metrics_frc = \
    compute_error_metrics(fnl_rel_aft, fnl_rel_nn)
global_metrics_frc_normalized, individual_metrics_frc_normalized = \
    compute_error_metrics(fnl_rel_aft, fnl_rel_nn, normalize=True)

# Spider plot of error metrics of FRC trajectory
labels = list(global_metrics_frc.keys())
values = [global_metrics_frc[k] for k in labels]
values_norm = [global_metrics_frc_normalized[k] for k in labels]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
values = np.r_[values, values[0]]
values_norm = np.r_[values_norm, values_norm[0]]
angles = np.r_[angles, angles[0]]
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
ax.set_yscale("log")
ax.plot(angles, values, marker="o", label="raw", color='#1D3557')
ax.fill(angles, values, alpha=0.2, color='#1D3557')
ax.plot(angles, values_norm, marker="o", label="normalized", color='#E63946')
ax.fill(angles, values_norm, alpha=0.2, color='#E63946')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.tick_params(axis='x', pad=12)
ax.set_ylim(0, max(values + values_norm)*1.1)
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)
plt.tight_layout()
plt.savefig('figures/error_metrics_spider_frc.svg', bbox_inches='tight')
plt.show()

# bar plot of individual error metrics per output of FRC trajectory
outputs = ["A1", "B1", "A3", "B3"]
colors = ['#1D3557', '#008b9a', '#f19699', '#e63946']
metrics = individual_metrics_frc.keys()
raw = np.vstack([individual_metrics_frc[k] for k in metrics])
norm = np.vstack([individual_metrics_frc_normalized[k] for k in metrics])
x = np.arange(len(metrics))
w = 0.18
fig, ax = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
for k, (title, data) in enumerate(zip(["raw", "normalized"], [raw, norm])):
    for j in range(len(outputs)):
        ax[k].bar(x + (j-1.5)*w, data[:, j], width=w, label=outputs[j],
                  color=colors[j])
    ax[k].set_xticks(x)
    ax[k].set_xticklabels(metrics)
    ax[k].set_title(title)
ax[0].set_ylabel("error (0 = better)")
ax[1].legend(title="output")
plt.tight_layout()
plt.savefig('figures/error_metrics_bar_frc.svg', bbox_inches='tight')
plt.show()

colors = ['#1D3557', '#008b9a', '#f19699', '#e63946']
q_labels = ['A1', 'B1', 'A3', 'B3']
fnl_labels = ['A1', 'B1', 'A3', 'B3']

# NN prediction vs. AFT ground-truth over test samples and FRC inputs
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
for i in range(4):
    ax[i].plot(fnl_test_aft[:, i], fnl_test_nn[:, i], 'o', label=fnl_labels[i],
               color='#A8DADC')
    ax[i].plot(fnl_rel_aft[:, i], fnl_rel_nn[:, i], 'x',
               label=f'{fnl_labels[i]} (rel)', color='#E63946')
    ax[i].set_xlabel(f'{fnl_labels[i]} true')
    ax[i].set_ylabel(f'{fnl_labels[i]} predicted')
    ax[i].legend()
plt.suptitle(f'ground truth vs. prediction for dataset {nn_id}')
plt.tight_layout()


# AFT vs. NN over FRC iterations
fig, ax = plt.subplots(4, 1, figsize=(5, 8))
for i in range(4):
    ax[0].plot(q_rel[:, i], label=q_labels[i], color=colors[i])
    ax[1].plot(fnl_rel_aft[:, i], label=fnl_labels[i],
               color=colors[i])
    ax[2].plot(fnl_rel_nn[:, i], label=f'{fnl_labels[i]}_NN', color=colors[i])
    ax[3].plot(fnl_rel_aft[:, i] - fnl_rel_nn[:, i],
               label=f'{fnl_labels[i]} error', color=colors[i])
ax[0].legend()
ax[0].set_title('Input coefficients over iterations')
ax[1].legend()
ax[1].set_title('AFT output coefficients over iterations')
ax[2].legend()
ax[2].set_title('NN output coefficients over iterations')
ax[3].legend()
ax[3].set_title('Error between AFT and NN outputs over iterations')
plt.tight_layout()

plt.show()
