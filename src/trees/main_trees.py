from src import data_utils, eval_utils

import os
import csv
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

RESULT_DIR = 'results/trees/'

RF_PARAMS = {'max_features': 'sqrt',
             'max_depth': None,
             'bootstrap': False
             }

DEFAULT_TREES = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_LEAVES = [2, 5, 10, 20, 50, 75, 100, 500, 1000, 2000, None]
DEFAULT_SEEDS = [42]


def run_forest_experiment(out_file_name='tree_part1_res',
                          dataset='MNIST',
                          n_tree_list: list = None,
                          n_leaves_list: list = None,
                          seed_list: list = None,
                          write_to_csv: bool = True,
                          return_res_frame: bool = False,
                          pred_idx='all',
                          store_ind_res: bool = True,
                          indep_regressors: bool = False):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if n_tree_list is None:
        n_tree_list = DEFAULT_TREES

    if n_leaves_list is None:
        n_leaves_list = DEFAULT_LEAVES

    if seed_list is None:
        seed_list = DEFAULT_SEEDS

    # dataframe to collect results
    header = ['seed',
              'n_estimators',
              'max_leaf_nodes',
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
        for n_trees in n_tree_list:
            for n_leaves in n_leaves_list:
                print(
                    "Running experiment with seed {}, n_estimators {} and max_leaf_nodes {}.".format(
                        seed, n_trees, n_leaves))

                if (not pred_idx == 'all') or (not indep_regressors):
                    # fit single forest, either multi-class or for single pred_idx
                    clf = RandomForestRegressor(n_estimators=n_trees,
                                                max_leaf_nodes=n_leaves,
                                                random_state=seed,
                                                **RF_PARAMS)
                    clf.fit(X_train, y_train)

                    # create predictions
                    y_pred_train = clf.predict(X_train)
                    y_pred_test = clf.predict(X_test)

                    # also compute metrics on individual o-v-a problems
                    if (pred_idx == 'all') and store_ind_res:
                        n_targets = y_test.shape[1]
                        for target in range(n_targets):
                            # store results for individual predictions
                            sse_train_target, err_train_target = \
                                eval_utils.compute_prediction_metrics(
                                    y_train,
                                    y_pred_train[
                                    :,
                                    target],
                                    pred_idx=target)
                            sse_test_target, err_test_target = \
                                eval_utils.compute_prediction_metrics(y_test,
                                                                      y_pred_test[
                                                                      :, target],
                                                                      pred_idx=target)
                            # save to frame
                            next_row = [seed, n_trees, n_leaves, err_train_target, err_test_target,
                                        sse_train_target, sse_test_target, target, dataset]

                            # write to file
                            if write_to_csv:
                                writer.writerow(next_row)

                            # write to frame
                            if return_res_frame:
                                new_frame = pd.DataFrame(columns=header,
                                                     data=[next_row])
                                out_frame = pd.concat([out_frame, new_frame])

                else:
                    # fit independent forests one at a time
                    n_targets = y_test.shape[1]
                    for target in range(n_targets):
                        clf = RandomForestRegressor(n_estimators=n_trees,
                                                    max_leaf_nodes=n_leaves,
                                                    random_state=seed,
                                                    **RF_PARAMS)
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
                            next_row = [seed, n_trees, n_leaves, err_train_target, err_test_target,
                                        sse_train_target, sse_test_target, target, dataset]

                            # write to file
                            if write_to_csv:
                                writer.writerow(next_row)

                            # write to frame
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

                # compute metrics on complete problem
                sse_train, err_train = eval_utils.compute_prediction_metrics(y_train, y_pred_train,
                                                                             pred_idx=pred_idx)
                sse_test, err_test = eval_utils.compute_prediction_metrics(y_test, y_pred_test,
                                                                           pred_idx=pred_idx)

                # save to frame
                next_row = [seed, n_trees, n_leaves, err_train, err_test,
                            sse_train, sse_test, pred_idx, dataset]

                # write to file
                if write_to_csv:
                    writer.writerow(next_row)

                # write to frame
                if return_res_frame:
                    new_frame = pd.DataFrame(columns=header,
                                         data=[next_row])
                    out_frame = pd.concat([out_frame, new_frame])

    if write_to_csv:
        out_file.close()

    if return_res_frame:
        return out_frame

if __name__ == '__main__':
    run_forest_experiment(out_file_name='tree_part1_res',
                          dataset='MNIST',
                          n_tree_list=[1, 2, 5, 10, 20, 50, 100],
                          n_leaves_list=[2, 5, 10, 20, 50, 75, 100, 500, 1000, 2000, None],
                          seed_list=[42],
                          write_to_csv=True,
                          return_res_frame=False,
                          pred_idx='all',
                          store_ind_res=True,
                          indep_regressors=False)
