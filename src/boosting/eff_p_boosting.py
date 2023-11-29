from src import data_utils, eff_p_utils

import os
import csv
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor

RESULT_DIR = 'results/boosting/'

BOOST_PARAMS = {'max_features': 'sqrt',
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'max_depth': None,
                'criterion': 'squared_error',
                'init': 'zero'
                }

BELKIN_SETTING = {'max_leaf_nodes': 10,
                  'learning_rate': .85}

DEFAULT_ENS = [1, 2, 5, 20, 50]
DEFAULT_BOOST = [1, 2, 10, 25, 50, 100, 200]
DEFAULT_SEEDS = [42]
DEFAULT_LR = [.85]
DEFAULT_LEAVES = [10]


def track_metrics_through_gbtreg_voter(vote_gbt, X_train, X_test, y_train, y_test,
                                       verbose=0):
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_est = len(vote_gbt.estimators_)

    # create frame to save
    header = ['n_ens_tot', 'n_ens',
              'train_error_mse', 'test_error_mse',
              'train_error_binary', 'test_error_binary',
              'eff_p_tr',
              'l2_train', 'l2_test',
              'l2_train_sq', 'l2_test_sq']

    out_frame = pd.DataFrame(columns=header)

    # initialise
    S_all_train = np.zeros(shape=(n_train, n_train))
    S_all_test = np.zeros(shape=(n_test, n_train))

    for n in range(n_est):
        if verbose > 0:
            print('Computing S for GB {}.'.format(n))
        S_train, S_test = eff_p_utils.create_S_from_gbtregressor(vote_gbt.estimators_[n], X_train,
                                                                 X_test)
        S_all_train += S_train
        S_all_test += S_test

        if verbose > 0:
            print('Computing metrics for GB {}.'.format(n))

        mse_train, mse_test, acc_train, acc_test, \
        eff_p_tr, l2_norm_train, \
        l2_norm_test, l2_norm_train_sq, l2_norm_test_sq = eff_p_utils.compute_metrics_from_S(
            S_all_train / S_all_train.sum(axis=1)[:, None],
            S_all_test / S_all_test.sum(axis=1)[:, None],
            y_train, y_test)

        # append data
        next_row = [n_est, n + 1, mse_train, mse_test,
                    acc_train, acc_test, eff_p_tr,
                    l2_norm_train, l2_norm_test, l2_norm_train_sq, l2_norm_test_sq,
                    ]

        new_frame = pd.DataFrame(columns=header,
                                 data=[next_row])
        out_frame = pd.concat([out_frame, new_frame])

    return out_frame


def track_metrics_through_single_gbtreg(gbtreg, X_train, X_test, y_train, y_test,
                                        verbose=0):
    # track change in metrics for single GBReg
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(gbtreg.estimators_)
    lr = gbtreg.learning_rate
    seed = gbtreg.random_state

    S_train_curr = np.zeros(shape=(n_train, n_train))
    S_test_curr = np.zeros(shape=(n_test, n_train))

    # create frame to save
    header = ['n_boost_tot', 'n_boost',
              'train_error_mse', 'test_error_mse',
              'train_error_binary', 'test_error_binary',
              'eff_p_tr',
              'l2_train', 'l2_test',
              'l2_train_sq', 'l2_test_sq']
    out_frame = pd.DataFrame(columns=header)

    for n in range(n_trees):
        if verbose > 0:
            print('Computing S for tree {}.'.format(n))
        S_train, S_test = eff_p_utils.create_S_from_single_boosted_tree(gbtreg.estimators_[n][0],
                                                                        None if n == 0 else S_train_curr,
                                                                        X_train, X_test)
        S_train_curr += lr * S_train
        S_test_curr += lr * S_test

        if verbose > 0:
            print('Computing metrics for tree {}.'.format(n))

        mse_train, mse_test, acc_train, acc_test, \
        eff_p_tr, l2_norm_train, \
        l2_norm_test, l2_norm_train_sq, l2_norm_test_sq = eff_p_utils.compute_metrics_from_S(
            S_train_curr,
            S_test_curr,
            y_train, y_test)

        # append data
        next_row = [n_trees, n + 1, mse_train, mse_test,
                    acc_train, acc_test, eff_p_tr,
                    l2_norm_train, l2_norm_test, l2_norm_train_sq, l2_norm_test_sq]

        new_frame = pd.DataFrame(columns=header,
                                 data=[next_row])
        out_frame = pd.concat([out_frame, new_frame])

    return out_frame


def run_eff_p_experiment_single_boost(out_file_name='eff_p_single_boost_res',
                                      dataset='MNIST',
                                      n_boost_max: int = 200,
                                      lr_list: list = None,
                                      n_leaves_list: list = None,
                                      seed_list: list = None,
                                      write_to_csv: bool = True,
                                      return_res_frame: bool = False,
                                      pred_idx=0,
                                      verbose=1):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if lr_list is None:
        lr_list = DEFAULT_LR

    if n_leaves_list is None:
        n_leaves_list = DEFAULT_LEAVES

    if seed_list is None:
        seed_list = DEFAULT_SEEDS

    header = ['pred_idx', 'seed', 'max_leaf_nodes', 'lr', 'dataset',
              'n_boost_tot', 'n_boost',
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
        for lr in lr_list:
            for n_leaves in n_leaves_list:
                print(
                    "Running part 2 experiment for boosting (single) with seed {}, learning rate "
                    "{} and max_leaf_nodes {}.".format(
                        seed, lr, n_leaves))

                clf = GradientBoostingRegressor(n_estimators=n_boost_max,
                                                max_leaf_nodes=n_leaves,
                                                learning_rate=lr,
                                                random_state=seed,
                                                **BOOST_PARAMS)

                clf.fit(X_train, y_train)

                out_iter = track_metrics_through_single_gbtreg(clf, X_train, X_test,
                                                               y_train,
                                                               y_test,
                                                               verbose)

                next_frame = pd.concat([
                    pd.DataFrame(
                        columns=['pred_idx', 'seed', 'max_leaf_nodes', 'lr', 'dataset'],
                        data=[[pred_idx, seed, n_leaves, lr, dataset]]
                    ),
                    out_iter
                ],
                    axis=1
                )

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


def run_eff_p_experiment_ensemble_boost(out_file_name='eff_p_ensemble_boost_res',
                                        dataset='MNIST',
                                        n_ens_max: int = 50,
                                        n_boost_list: list = None,
                                        seed_list: list = None,
                                        write_to_csv: bool = True,
                                        return_res_frame: bool = False,
                                        pred_idx=0,
                                        verbose=1):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if n_boost_list is None:
        n_boost_list = DEFAULT_BOOST

    if seed_list is None:
        seed_list = DEFAULT_SEEDS

    header = ['pred_idx', 'seed', 'n_boost', 'dataset',
              'n_ens_tot',  'n_ens',
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
        for n_boost in n_boost_list:
            print('Running part 2 experiment for boosting (voting) with seed {} and n_boost {'
                  '}'.format(seed, n_boost))

            est_list = []
            for member in range(n_ens_max):
                est_list.append(
                    (
                        'est_' + str(member),
                        GradientBoostingRegressor(
                            n_estimators=n_boost,
                            random_state=seed * (member + 1),
                            **BOOST_PARAMS, **BELKIN_SETTING
                        )
                    )
                )
            clf = VotingRegressor(est_list)

            clf.fit(X_train, y_train)

            out_iter = track_metrics_through_gbtreg_voter(clf, X_train, X_test, y_train, y_test,
                                                          verbose)

            next_frame = pd.concat(
                [
                    pd.DataFrame(
                        columns=['pred_idx', 'seed', 'n_boost', 'dataset'],
                        data=[[pred_idx, seed, n_boost, dataset]]
                    ),
                    out_iter
                ],
                axis=1)

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
    # run_eff_p_experiment_single_boost()
    run_eff_p_experiment_ensemble_boost()
