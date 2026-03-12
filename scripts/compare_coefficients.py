import numpy as np
from src.aft import compute_AFT_solution
from src.fourier_conversion import (convert_cossin_to_comexp,
                                    convert_comexp_to_cossin)
from src.nn_inference import evaluate_Duffing_nn_H3
from src.error_metrics import compute_error_metrics
from src.plotting import (error_metrics_spider_plot,
                          individual_error_metrics_bar_plot,
                          individual_normalized_mse_bar_plot,
                          coefficients_over_iterations_plot,
                          prediciton_vs_ground_truth_plot,
                          all_predictions_vs_ground_truths_inset_plot)


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

error_metrics_spider_plot(global_metrics_test, global_metrics_test_normalized,
                          figure_name='error_metrics_spider_test')

individual_error_metrics_bar_plot(individual_metrics_test,
                                  individual_metrics_test_normalized,
                                  figure_name='error_metrics_bar_test')

individual_normalized_mse_bar_plot(individual_metrics_test_normalized,
                                   figure_name='normalized_mse_bar_test')

all_predictions_vs_ground_truths_inset_plot(
    [fnl_test_aft],
    [fnl_test_nn],
    figure_name='all_predictions_vs_ground_truths_test')


###############################################################################
# Performance on FRC trajectory
###############################################################################
q_frc_full = np.loadtxt('data/input_frc_aft.txt', delimiter=',')
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

error_metrics_spider_plot(global_metrics_frc, global_metrics_frc_normalized,
                          figure_name='error_metrics_spider_frc')

individual_error_metrics_bar_plot(individual_metrics_frc,
                                  individual_metrics_frc_normalized,
                                  figure_name='error_metrics_bar_frc')

all_predictions_vs_ground_truths_inset_plot(
    [fnl_rel_aft],
    [fnl_rel_nn],
    figure_name='all_predictions_vs_ground_truths_frc')

# AFT vs. NN over FRC iterations
coefficients_over_iterations_plot(q_rel, fnl_rel_aft, fnl_rel_nn)


###############################################################################
# NN prediction vs. AFT ground-truth over test samples and FRC inputs
###############################################################################
prediciton_vs_ground_truth_plot([fnl_test_aft, fnl_rel_aft],
                                [fnl_test_nn, fnl_rel_nn])
