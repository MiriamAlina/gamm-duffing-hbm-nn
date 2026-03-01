import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error
    )
from src.cosine_similarity import cosine_similarity
from src.aft import compute_AFT_solution
from src.fourier_conversion import (convert_cossin_to_comexp,
                                    convert_comexp_to_cossin)
from src.nn_inference import evaluate_Duffing_nn_H3


def compare_coefficients(y_true, y_pred, normalize=False, eps=1e-14):
    """
    Compare two arrays of shape (N, 4) using common regression metrics.
    
    Parameters
    ----------
    y_true : ndarray (N,4)
    y_pred : ndarray (N,4)
    normalize : bool
        If True, normalize using mean/std of y_true per output.
    """
    
    if normalize:
        mean = y_true.mean(axis=0, keepdims=True)
        std = y_true.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        y_true = (y_true - mean) / std
        y_pred = (y_pred - mean) / std

    assert y_true.shape == y_pred.shape
    N, d = y_true.shape

    # ---- Global metrics (averaged over outputs) ----
    rel_l2_global = np.linalg.norm(y_true - y_pred) / \
                    (np.linalg.norm(y_true) + eps)  # global relative L2 error (spectral energy error)
    r2_global = r2_score(y_true, y_pred, multioutput='uniform_average')
    mse_global = mean_squared_error(y_true, y_pred, multioutput='uniform_average')
    mae_global = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
    rmse_global = np.sqrt(mse_global)

    # cosine similarity over whole dataset (flattened vectors)
    cos_global = cosine_similarity(
        y_true.ravel(),
        y_pred.ravel()
    )

    # ---- Per-output metrics ----
    rel_l2_per_mode = []
    r2_per = []
    mse_per = []
    mae_per = []
    rmse_per = []
    cos_per = []

    for i in range(d):
        err = np.linalg.norm(y_true[:, i] - y_pred[:, i])
        norm = np.linalg.norm(y_true[:, i]) + eps
        rel_l2_per_mode.append(err / norm)
        r2_per.append(r2_score(y_true[:, i], y_pred[:, i]))
        mse_i = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse_i = np.sqrt(mse_i)

        cos_i = cosine_similarity(
            y_true[:, i].ravel(),
            y_pred[:, i].ravel()
        )

        mse_per.append(mse_i)
        mae_per.append(mae_i)
        rmse_per.append(rmse_i)
        cos_per.append(cos_i)

    rel_l2_balanced = np.mean(rel_l2_per_mode)

    print("========== GLOBAL METRICS ==========")
    print(f"rel L2 error    : {rel_l2_global:.6e}")
    print(f"R²              : {r2_global:.6f}")
    print(f"MSE             : {mse_global:.6e}")
    print(f"RMSE            : {rmse_global:.6e}")
    print(f"MAE             : {mae_global:.6e}")
    print(f"CosSim          : {cos_global:.6f}")

    print("\n====== PER OUTPUT METRICS ======")
    for i in range(d):
        print(f"\nOutput {i}:")
        print(f"Balanced mean relative L2 error : {rel_l2_balanced:.6e}")
        print(f"  R²     : {r2_per[i]:.6f}")
        print(f"  MSE    : {mse_per[i]:.6e}")
        print(f"  RMSE   : {rmse_per[i]:.6e}")
        print(f"  MAE    : {mae_per[i]:.6e}")
        print(f"  CosSim : {cos_per[i]:.6f}")

    return {
        "rel_l2_global": rel_l2_global,
        "rel_l2_balanced": rel_l2_balanced,
        "rel_l2_per_mode": rel_l2_per_mode,
        "cos_global": cos_global,
        "r2_per_mode": r2_per,
        "mse_per_mode": mse_per
    }


def metrics_to_radar_scores(metrics):
    scores = {}

    # ----- Errors → 1/(1+error) -----
    scores["rel_L2"] = 1 / (1 + metrics["rel_l2_global"])

    mse_mean = np.mean(metrics["mse_per_mode"])
    scores["MSE"] = 1 / (1 + mse_mean)

    # RMSE nicht separat gespeichert → ableiten
    rmse_mean = np.sqrt(mse_mean)
    scores["RMSE"] = 1 / (1 + rmse_mean)

    # MAE nicht gespeichert → optional ergänzen
    # falls du willst, füge MAE_mean oben in return hinzu

    # ----- R² -----
    r2_mean = np.mean(metrics["r2_per_mode"])
    scores["R2"] = max(0.0, r2_mean)

    # ----- Cosine -----
    scores["CosSim"] = 0.5 * (1 + metrics["cos_global"])

    return scores



def plot_radar(scores):

    labels = list(scores.keys())
    values = list(scores.values())

    # schließen des Polygons
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0,1)

    plt.show()


# training data
training_data_id = '2026-02-18_14-04-47'
training_data = np.load('data/duffing_training_data_H3_N64_' +
                        training_data_id + '.npz')
q_train = training_data['q_coeffs']
fnl_train_aft = training_data['fnl_coeffs']

# NN performance on training data
nn_id = '2026-02-18_13-29-30'  # identifier of the trained NN model
fnl_train_nn = []
for i in range(len(q_train)):
    q_input = np.concatenate([q_train[i, :]])
    fnl_train_nn.append(evaluate_Duffing_nn_H3(nn_id, q_input))
fnl_train_nn = np.array(fnl_train_nn)


# physically relevant inputs over FRC iterations
q_rel = np.loadtxt('data/nn_input_Duffing.txt', delimiter=',')

# NN performance on FRC relevant input trajectory
H = 3
N = 2**6
gamma = 0.1
fnl_rel_aft = []
fnl_rel_nn = []
for i in range(np.shape(q_rel)[0]):
    q_ce = convert_cossin_to_comexp(q_rel[i, :7])
    fnl_ce = compute_AFT_solution(N, H, q_ce, gamma)
    fnl_cs = convert_comexp_to_cossin(fnl_ce, H)
    fnl_cs_NN = evaluate_Duffing_nn_H3(
        nn_id, np.concatenate([q_rel[i, 1:3], q_rel[i, 5:7]]))
    fnl_rel_aft.append(fnl_cs)
    fnl_rel_nn.append(fnl_cs_NN)

fnl_rel_aft = np.array(fnl_rel_aft)
fnl_rel_aft_comparison = fnl_rel_aft[:, [1, 2, 5, 6]]
fnl_rel_nn = np.array(fnl_rel_nn)
metrics = compare_coefficients(fnl_rel_aft_comparison, fnl_rel_nn)
radar_scores = metrics_to_radar_scores(metrics)
plot_radar(radar_scores)


# NN prediction vs. AFT ground-truth over training samples and FRC inputs
fig, ax = plt.subplots(2, 2, figsize=(7, 8))
ax[0, 0].plot(fnl_train_aft[:, 0], fnl_train_nn[:, 0], 'o', label='A1',
              color='#A8dadc')
ax[0, 0].plot(fnl_rel_aft[:, 1], fnl_rel_nn[:, 0], 'x',
              label='A1 (rel)', color='#E63946')
ax[0, 0].set_xlabel('A1 true')
ax[0, 0].set_ylabel('A1 predicted')
ax[0, 0].legend()
ax[0, 1].plot(fnl_train_aft[:, 1], fnl_train_nn[:, 1], 'o', label='B1',
              color='#A8dadc')
ax[0, 1].plot(fnl_rel_aft[:, 2], fnl_rel_nn[:, 1], 'x',
              label='B1 (rel)', color='#E63946')
ax[0, 1].set_xlabel('B1 true')
ax[0, 1].set_ylabel('B1 predicted')
ax[0, 1].legend()
ax[1, 0].plot(fnl_train_aft[:, 2], fnl_train_nn[:, 2], 'o', label='A3',
              color='#A8dadc')
ax[1, 0].plot(fnl_rel_aft[:, 5], fnl_rel_nn[:, 2], 'x',
              label='A3 (rel)', color='#E63946')
ax[1, 0].set_xlabel('A3 true')
ax[1, 0].set_ylabel('A3 predicted')
ax[1, 0].legend()
ax[1, 1].plot(fnl_train_aft[:, 3], fnl_train_nn[:, 3], 'o', label='B3',
              color='#A8dadc')
ax[1, 1].plot(fnl_rel_aft[:, 6], fnl_rel_nn[:, 3], 'x',
              label='B3 (rel)', color='#E63946')
ax[1, 1].set_xlabel('B3 true')
ax[1, 1].set_ylabel('B3 predicted')
ax[1, 1].legend()
plt.suptitle(f'ground truth vs. prediction for dataset {nn_id}')
plt.tight_layout()


# AFT vs. NN over FRC iterations
colors = ['#1D3557', '#008b9a', '#f19699', '#e63946']
fig, ax = plt.subplots(4, 1, figsize=(5, 8))
ax[0].plot(q_rel[:, 1], label='a1', color=colors[0])
ax[0].plot(q_rel[:, 2], label='b1', color=colors[1])
ax[0].plot(q_rel[:, 5], label='a3', color=colors[2])
ax[0].plot(q_rel[:, 6], label='b3', color=colors[3])
ax[0].legend()
ax[0].set_title('Input coefficients over iterations')
ax[1].plot(fnl_rel_aft[:, 1], label='A1', color=colors[0])
ax[1].plot(fnl_rel_aft[:, 2], label='B1', color=colors[1])
ax[1].plot(fnl_rel_aft[:, 5], label='A3', color=colors[2])
ax[1].plot(fnl_rel_aft[:, 6], label='B3', color=colors[3])
ax[1].legend()
ax[1].set_title('AFT output coefficients over iterations')
ax[2].plot(fnl_rel_nn[:, 0], label='A1_NN', color=colors[0])
ax[2].plot(fnl_rel_nn[:, 1], label='B1_NN', color=colors[1])
ax[2].plot(fnl_rel_nn[:, 2], label='A3_NN', color=colors[2])
ax[2].plot(fnl_rel_nn[:, 3], label='B3_NN', color=colors[3])
ax[2].legend()
ax[2].set_title('NN output coefficients over iterations')
ax[3].plot(fnl_rel_aft[:, 1]-fnl_rel_nn[:, 0], label='A1 error',
           color=colors[0])
ax[3].plot(fnl_rel_aft[:, 2]-fnl_rel_nn[:, 1], label='B1 error',
           color=colors[1])
ax[3].plot(fnl_rel_aft[:, 5]-fnl_rel_nn[:, 2], label='A3 error',
           color=colors[2])
ax[3].plot(fnl_rel_aft[:, 6]-fnl_rel_nn[:, 3], label='B3 error',
           color=colors[3])
ax[3].legend()
ax[3].set_title('Error between AFT and NN outputs over iterations')
plt.tight_layout()

plt.show()
