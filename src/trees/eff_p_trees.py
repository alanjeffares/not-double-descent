from src import data_utils, eff_p_utils

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

DEFAULT_TREES = [100]
DEFAULT_LEAVES = [2, 5, 10, 20, 50, 75, 100, 500, 1000, 2000, None]
DEFAULT_SEEDS = [42]


def track_metrics_through_single_forest(forest, X_train, X_test, y_train, y_test,
                                        verbose=0):
    # track change in metrics for single forest as we loop through estimators

    header = ['n_trees_tot',  'n_estimators',
                  'train_error_mse', 'test_error_mse',
                  'train_error_binary', 'test_error_binary',
              'eff_p_tr',
              'l2_train', 'l2_test',
              'l2_train_sq', 'l2_test_sq']

    # create frame to save results in
    out_frame = pd.DataFrame(columns=header)

    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(forest.estimators_)

    S_train_curr = np.zeros(shape=(n_train, n_train))
    S_test_curr = np.zeros(shape=(n_test, n_train))

    for n in range(n_trees):
        if verbose > 0:
            print('Computing S for forest with {} trees.'.format(n + 1))
        S_train, S_test = eff_p_utils.create_S_from_tree(forest.estimators_[n], X_train, X_test)
        S_train_curr += S_train
        S_test_curr += S_test

        if verbose > 0:
            print('Computing metrics for forest with {} trees.'.format(n + 1))

        mse_train, mse_test, acc_train, acc_test, \
        eff_p_tr, l2_norm_train, \
        l2_norm_test, l2_norm_train_sq, l2_norm_test_sq = eff_p_utils.compute_metrics_from_S(
            S_train_curr / S_train_curr.sum(axis=1)[:, None],
            S_test_curr / S_test_curr.sum(axis=1)[:, None],
            y_train, y_test)

        # append data
        next_row = [n_trees, n + 1, mse_train, mse_test,
                    acc_train, acc_test, eff_p_tr,
                    l2_norm_train, l2_norm_test,
                    l2_norm_train_sq, l2_norm_test_sq]

        new_frame = pd.DataFrame(columns=header,
                                 data=[next_row])
        out_frame = pd.concat([out_frame, new_frame])

    return out_frame


def run_eff_p_experiment_trees(out_file_name='eff_p_tree_res',
                               dataset='MNIST',
                               n_tree_list: list = None,
                               n_leaves_list: list = None,
                               seed_list: list = None,
                               write_to_csv: bool = True,
                               return_res_frame: bool = False,
                               pred_idx=0,
                               verbose=1):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if n_tree_list is None:
        n_tree_list = DEFAULT_TREES

    if n_leaves_list is None:
        n_leaves_list = DEFAULT_LEAVES

    if seed_list is None:
        seed_list = DEFAULT_SEEDS


    header = ['pred_idx', 'seed', 'max_leaf_nodes', 'dataset',
                  'n_trees_tot', 'n_estimators',
                  'train_error_mse', 'test_error_mse',
                  'train_error_binary', 'test_error_binary',
                  'eff_p_tr',
                  'l2_train', 'l2_test',
                  'l2_train_sq', 'l2_test_sq']

    if write_to_csv:
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        if not os.path.isfile(RESULT_DIR + (out_file_name + ".csv")):
            # if file does not exist yet
            out_file = open(RESULT_DIR + (out_file_name + ".csv"), "w", newline='', buffering=1)
            writer = csv.writer(out_file)
            writer.writerow(header)
        else:
            # just open existing file
            out_file = open(RESULT_DIR + (out_file_name + ".csv"), "a", newline='', buffering=1)
            writer = csv.writer(out_file)

    # create frame to save results in
    if return_res_frame:
        out_frame = pd.DataFrame(columns=header)

    X_train, y_train, X_test, y_test = data_utils.get_data(dataset=dataset, pred_idx=pred_idx,
                                                           center=True)

    for seed in seed_list:
        for n_trees in n_tree_list:
            for n_leaves in n_leaves_list:
                print(
                    "Running experiment with seed {}, n_estimators {} and max_leaf_nodes {}.".format(
                        seed, n_trees, n_leaves))

                clf = RandomForestRegressor(n_estimators=n_trees,
                                            max_leaf_nodes=n_leaves,
                                            random_state=seed,
                                            **RF_PARAMS)
                clf.fit(X_train, y_train)

                out_iter = track_metrics_through_single_forest(clf, X_train, X_test, y_train,
                                                               y_test, verbose=verbose)


                next_frame = pd.concat([pd.DataFrame(columns=['pred_idx', 'seed',
                                                              'max_leaf_nodes', 'dataset'],
                                                         data=[[pred_idx, seed, n_leaves, dataset]]),
                                        out_iter], axis=1)
                if return_res_frame:
                    out_frame = pd.concat([out_frame, next_frame])

                if write_to_csv:
                    for i in range(next_frame.shape[0]):
                        next_row = next_frame.iloc[i, :].values
                        writer.writerow(next_row)

    if write_to_csv:
        out_file.close()

    if return_res_frame:
        return out_frame

if __name__ == '__main__':
    run_eff_p_experiment_trees(out_file_name='eff_p_tree_res',
                               dataset='MNIST',
                               n_tree_list=DEFAULT_TREES,
                               n_leaves_list=DEFAULT_LEAVES,
                               seed_list=DEFAULT_SEEDS,
                               write_to_csv=True,
                               return_res_frame=False,
                               pred_idx='all',
                               verbose=1)