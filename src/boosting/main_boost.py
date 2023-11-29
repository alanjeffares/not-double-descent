from src import data_utils, eval_utils

import os
import csv
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor

RESULT_DIR = 'results/boosting/'

BOOST_PARAMS = {'max_features': 'sqrt',
                "max_leaf_nodes": 10,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'learning_rate': 0.85,
                'max_depth': None,
                'criterion': 'squared_error',
                'init': 'zero'
                }

DEFAULT_ENS = [1, 2, 5, 10, 20]
DEFAULT_BOOST = [1, 2, 10, 25, 50, 100, 200]
DEFAULT_SEEDS = [42]


def run_boosting_experiment(out_file_name='boost_part1_res',
                          dataset='MNIST',
                            n_ens_list: list = None,
                            n_boost_list: list = None,
                            seed_list: list = None,
                            write_to_csv: bool = True,
                            return_res_frame: bool = False,
                            pred_idx='all',
                            store_ind_res: bool = True):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if n_ens_list is None:
        n_ens_list = DEFAULT_ENS

    if n_boost_list is None:
        n_boost_list = DEFAULT_BOOST

    if seed_list is None:
        seed_list = DEFAULT_SEEDS

    # dataframe to collect results
    header = ['seed',
              'n_ens',
              'n_boost',
              'train_error_binary',
              'test_error_binary',
              'train_error_mse',
              'test_error_mse',
              'pred_idx',
              'dataset'
              ]

    if return_res_frame:
        out_frame = pd.DataFrame(columns=header)

    if write_to_csv:
        if not os.path.isfile(RESULT_DIR + (out_file_name + ".csv")):
            # if file does not exist yet
            out_file = open(RESULT_DIR + (out_file_name + ".csv"), "w", newline='', buffering=1)
            writer = csv.writer(out_file)
            writer.writerow(header)
        else:
            # just open existing file
            out_file = open(RESULT_DIR + (out_file_name + ".csv"), "a", newline='', buffering=1)
            writer = csv.writer(out_file)

    X_train, y_train, X_test, y_test = data_utils.get_data(dataset=dataset, pred_idx=pred_idx)

    for seed in seed_list:
        for n_ens in n_ens_list:
            for n_boost in n_boost_list:
                print("Running experiment with seed {},  {} ensemble members and {} "
                      "boosting rounds.".format(
                    seed, n_ens, n_boost))
                if pred_idx == 'all':
                    n_targets = y_test.shape[1]
                    for target in range(n_targets):
                        if n_ens == 1:
                            clf = GradientBoostingRegressor(n_estimators=n_boost,
                                                            random_state=seed, **BOOST_PARAMS)
                        else:
                            est_list = []
                            for member in range(n_ens):
                                est_list.append((
                                    'est_' + str(member),
                                    GradientBoostingRegressor(
                                        n_estimators=n_boost,
                                        random_state=seed * (member + 1),
                                        **BOOST_PARAMS)
                                )
                                )
                            clf = VotingRegressor(est_list)

                        clf.fit(X_train, y_train[:, target])

                        y_pred_train_target = clf.predict(X_train)
                        y_pred_test_target = clf.predict(X_test)

                        if store_ind_res:
                            # store results for individual predictions
                            sse_train_target, err_train_target = \
                                eval_utils.compute_prediction_metrics(
                                    y_train,
                                    y_pred_train_target,
                                    pred_idx=target)
                            sse_test_target, err_test_target = \
                                eval_utils.compute_prediction_metrics(y_test,
                                                                      y_pred_test_target,
                                                                      pred_idx=target)
                            # save to frame
                            next_row = [seed, n_ens, n_boost, err_train_target, err_test_target,
                                        sse_train_target, sse_test_target, target, dataset]

                            # write to file
                            if write_to_csv:
                                writer.writerow(next_row)

                            if return_res_frame:
                                new_frame = pd.DataFrame(columns=header,
                                                         data=[next_row])
                                out_frame = pd.concat([out_frame, new_frame])

                        # collect predictions for full problem
                        if target == 0:
                            y_pred_train = y_pred_train_target.reshape((-1, 1))
                            y_pred_test = y_pred_test_target.reshape((-1, 1))
                        else:
                            y_pred_train = np.concatenate(
                                [y_pred_train, y_pred_train_target.reshape((-1, 1))],
                                axis=1)
                            y_pred_test = np.concatenate(
                                [y_pred_test, y_pred_test_target.reshape((-1, 1))],
                                axis=1)

                # evaluate predictions on joint problem
                sse_train, err_train = eval_utils.compute_prediction_metrics(y_train, y_pred_train,
                                                                             pred_idx=pred_idx)
                sse_test, err_test = eval_utils.compute_prediction_metrics(y_test, y_pred_test,
                                                                           pred_idx=pred_idx)

                # save to frame
                next_row = [seed, n_ens, n_boost, err_train, err_test,
                            sse_train, sse_test, pred_idx, dataset]

                # write to file
                if write_to_csv:
                    writer.writerow(next_row)

                if return_res_frame:
                    new_frame = pd.DataFrame(columns=header,
                                             data=[next_row])
                    out_frame = pd.concat([out_frame, new_frame])

    if write_to_csv:
        out_file.close()

    if return_res_frame:
        return out_frame


if __name__ == '__main__':
    run_boosting_experiment(out_file_name='boost_part1_res',
                          dataset='MNIST',
                            n_ens_list=[1, 2, 5, 10, 20],
                            n_boost_list=[1, 2, 10, 25, 50, 100, 200],
                            seed_list=[42],
                            write_to_csv=True,
                            return_res_frame=False,
                            pred_idx='all',
                            store_ind_res=True)