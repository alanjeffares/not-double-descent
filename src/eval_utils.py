import numpy as np
from sklearn.metrics import mean_squared_error


def compute_prediction_metrics(y_true, y_pred, pred_idx='all', threshold=.5):
    if pred_idx == 'all':
        # compute sum of squared errors
        n_targets = y_pred.shape[1]
        squared_errs = []
        for target in range(n_targets):
            squared_errs.append(mean_squared_error(y_true[:, target], y_pred[:, target]))
        tot_squared_error = np.sum(squared_errs)

        # compute error percentage
        err_pc = np.mean(np.argmax(y_pred, axis=1) != np.argmax(y_true, axis=1))

    else:
        tot_squared_error = mean_squared_error(y_true[:, pred_idx], y_pred)
        err_pc = np.mean((y_pred > threshold).astype(float) != y_true[:, pred_idx])

    return tot_squared_error, err_pc