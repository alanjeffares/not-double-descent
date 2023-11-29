import numpy as np
from sklearn.metrics import mean_squared_error


def create_S_from_tree(tree, X_train, X_test):
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # get node labels
    nodes_train = tree.apply(X_train).reshape((n_train, 1))
    nodes_test = tree.apply(X_test).reshape((n_test, 1))

    # expand to vectorize S creation
    nodes_train_exp = np.repeat(nodes_train, n_train, axis=1)
    nodes_test_exp = np.repeat(nodes_train, n_test, axis=1)

    # create S on training data
    S_train = np.transpose(nodes_train_exp) == nodes_train
    S_train = S_train / S_train.sum(axis=1)[:, None]

    # create S on test data
    S_test = np.transpose(nodes_test_exp) == nodes_test
    S_test = S_test / S_test.sum(axis=1)[:, None]

    return S_train, S_test


def create_S_from_forest(forest, X_train, X_test):
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(forest.estimators_)

    # initialise
    S_all_train = np.zeros(shape=(n_train, n_train))
    S_all_test = np.zeros(shape=(n_test, n_train))
    for n in range(n_trees):
        S_train, S_test = create_S_from_tree(forest.estimators_[n], X_train, X_test)
        S_all_train += S_train
        S_all_test += S_test
    S_all_train = S_all_train / S_all_train.sum(axis=1)[:, None]
    S_all_test = S_all_test / S_all_test.sum(axis=1)[:, None]

    return S_all_train, S_all_test


def create_S_from_single_boosted_tree(tree, S_gb_prev, X_train, X_test):
    S_train, S_test = create_S_from_tree(tree, X_train, X_test)
    if S_gb_prev is None:
        # first tree: just normal tree
        return S_train, S_test

    # reduce contributions where appropriate ----
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # get nodes in tree
    nodes_train = tree.apply(X_train).reshape((n_train, 1))
    nodes_test = tree.apply(X_test).reshape((n_test, 1))

    # get unique nodes
    all_nodes = np.unique(nodes_train)
    n_nodes = len(all_nodes)

    node_corrections = np.zeros(shape=(n_nodes, n_train))
    S_train_correction = np.zeros(shape=(n_train, n_train))
    S_test_correction = np.zeros(shape=(n_test, n_train))

    for i, n in enumerate(all_nodes):
        # create correction matrix
        leaf_id_train = (nodes_train == n).reshape((-1,))
        node_corrections[i, :] = S_gb_prev[leaf_id_train, :].sum(axis=0) / np.sum(leaf_id_train)

        # collect for train examples
        S_train_correction[leaf_id_train, :] = node_corrections[i, :]

        # collect for test examples
        leaf_id_test = (nodes_test == n).reshape((-1,))
        S_test_correction[leaf_id_test, :] = node_corrections[i, :]

    S_train = S_train - S_train_correction
    S_test = S_test - S_test_correction
    return S_train, S_test


def create_S_from_gbtregressor(gbtreg, X_train, X_test, verbose=0):
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(gbtreg.estimators_)
    lr = gbtreg.learning_rate

    S_train_curr = np.zeros(shape=(n_train, n_train))
    S_test_curr = np.zeros(shape=(n_test, n_train))

    for n in range(n_trees):
        if verbose > 0:
            print('Computing S for tree {}.'.format(n))
        S_train, S_test = create_S_from_single_boosted_tree(gbtreg.estimators_[n][0],
                                                            None if n == 0 else S_train_curr,
                                                            X_train, X_test)
        S_train_curr += lr * S_train
        S_test_curr += lr * S_test

    return S_train_curr, S_test_curr


def compute_metrics_from_S(S_train, S_test, y_train, y_test):
    # compute predictions
    y_pred_train = np.matmul(S_train, y_train)
    y_pred_test = np.matmul(S_test, y_train)

    # compute MSE and missclassification
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    acc_train = 1 - np.mean((y_pred_train > 0) == (y_train > 0))
    acc_test = 1 - np.mean((y_pred_test > 0) == (y_test > 0))

    # compute trace metric
    eff_p_tr = np.trace(S_train)

    # compute l2-norm
    l2_norm_train = np.mean(np.linalg.norm(S_train, axis=1))
    l2_norm_test = np.mean(np.linalg.norm(S_test, axis=1))

    l2_norm_train_sq = np.mean(np.linalg.norm(S_train, axis=1) ** 2)
    l2_norm_test_sq = np.mean(np.linalg.norm(S_test, axis=1) ** 2)

    return mse_train, mse_test, acc_train, acc_test, eff_p_tr, l2_norm_train, l2_norm_test, l2_norm_train_sq, l2_norm_test_sq
