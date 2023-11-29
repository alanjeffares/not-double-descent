import utils
import models
import torch
from typing import List


def fit_all(models: List[models.LinearRegression], X: torch.Tensor, y: torch.Tensor):
    """Fit all models in a one-vs-all fashion"""
    for target in range(10): # for each class
        y_ova = utils.ova_labels(target, y)
        models[target].fit(X, y_ova)
    return models

def eval_all(models: List[models.LinearRegression], X: torch.Tensor, y: torch.Tensor):
    """Predict and evaluate all models in a one-vs-all fashion"""
    for target in range(10):
        y_pred = models[target].predict(X)
        if target == 0:
            y_pred_all = y_pred.unsqueeze(1)
        else:
            y_pred_all = torch.cat((y_pred_all, y_pred.unsqueeze(1)), dim=1)
    y_pred = torch.argmax(y_pred_all, dim=1)
    error = (y_pred != y).sum()/len(y_pred)
    return error

def eval_ind_sqr_all(LR_list: List[models.LinearRegression], X: torch.Tensor, y: torch.Tensor):
    """Evaluate individual squared errors for all 10 models in a one-vs-all fashion"""
    errors_ova = []
    for target in range(10):
        y_ova = utils.ova_labels(target, y)
        y_pred = LR_list[target].predict(X)
        # calculate squared error 
        squared_error = ((y_pred - y_ova)**2).sum()/len(y_pred)
        errors_ova.append(squared_error.item())
    return errors_ova

# caluculate effective parameters on train set for all models
def eff_params_all(LR_list: List[models.LinearRegression], X: torch.Tensor):
    """Calculate effective parameters for all models in a one-vs-all fashion"""
    # note: all models have the same effective parameters since it is calculated
    # as a function of just X, not X and y. Therefore we only need to calculate
    # the effective parameters for a single of the ten models.
    eff_p_l2_ova = LR_list[0].eff_p_l2(X).item()
    eff_p_l2_squared_ova = LR_list[0].eff_p_l2_squared(X).item()
    return eff_p_l2_squared_ova, eff_p_l2_ova