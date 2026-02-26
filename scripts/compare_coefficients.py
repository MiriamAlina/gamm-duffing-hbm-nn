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


def compare_coefficients(AFT_result, NN_result, description):
    '''
    Evaluates the difference of two vectors according to different error
    metrics.
    Input: two coefficient vectors in cosine-sine form (real vectors)
    '''

    r2_result = r2_score(AFT_result, NN_result)
    mse_result = mean_squared_error(AFT_result, NN_result)
    mae_result = mean_absolute_error(AFT_result, NN_result)
    rmse_result = root_mean_squared_error(AFT_result, NN_result)
    cossim_result = cosine_similarity(AFT_result, NN_result)

    print(f"R^2 Score\t\t\t(best:1): {np.round(r2_result, 4)}")
    print(f"Mean Squared Error\t\t(best 0): {np.round(mse_result, 4)}")
    print(f"Mean Absolute Error\t\t(best 0): {np.round(mae_result, 4)}")
    print(f"Root Mean Squared Error\t\t(best 0): {np.round(rmse_result, 4)}")
    print(f"Cosine Similarity\t\t(best 1): {np.round(cossim_result, 4)}")

    return


def compare_test_set(AFT_results, NN_results):
    mean = AFT_results.mean(axis=0, keepdims=True)
    std = AFT_results.std(axis=0, keepdims=True)
    AFT_results_normalized = (AFT_results - mean) / std
    NN_results_normalized = (NN_results - mean) / std

    r2_per_output = [
        r2_score(AFT_results_normalized[:, i], NN_results_normalized[:, i])
        for i in range(AFT_results.shape[1])
    ]

    r2_mean = np.mean(r2_per_output)

    print("normalized R^2 Score per output (whole test set): " +
          f"{np.round(r2_per_output, 4)}" + "}")
    print("normalized mean R^2 Score (whole test set):" +
          str(np.round(r2_mean, 4)))

    return


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
fnl_rel_nn = np.array(fnl_rel_nn)


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
