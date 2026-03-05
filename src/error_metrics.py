import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error
    )
from src.cosine_similarity import cosine_similarity


def compute_error_metrics(y_true, y_pred, normalize=False, eps=1e-14):
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
    mae_global = mean_absolute_error(y_true, y_pred,
                                     multioutput='uniform_average')
    mse_global = mean_squared_error(y_true, y_pred,
                                    multioutput='uniform_average')
    rmse_global = root_mean_squared_error(y_true, y_pred,
                                          multioutput='uniform_average')
    cos_global = cosine_similarity(y_true.ravel(), y_pred.ravel())
    r2_global = r2_score(y_true, y_pred, multioutput='uniform_average')
    rel_l2_global = (np.linalg.norm(y_true - y_pred) /
                     (np.linalg.norm(y_true) + eps))

    # ---- Per-output metrics ----
    mae_individual = []
    mse_individual = []
    rmse_individual = []
    r2_individual = []
    for i in range(d):
        mae_individual.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        mse_individual.append(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rmse_individual.append(root_mean_squared_error(y_true[:, i],
                                                       y_pred[:, i]))
        r2_individual.append(r2_score(y_true[:, i], y_pred[:, i]))

    global_error_metrics = {
        "MAE": mae_global,
        "MSE": mse_global,
        "RMSE": rmse_global,
        "Cosine\nDistance": 1-cos_global,
        r"1-R$^2$": 1-r2_global,
        "Relative\n"+r"L$^2$ norm": rel_l2_global}
    individual_error_metrics = {
        "MAE": mae_individual,
        "MSE": mse_individual,
        "RMSE": rmse_individual,
        r"1-R$^2$": [1 - r2 for r2 in r2_individual]
    }

    return global_error_metrics, individual_error_metrics
