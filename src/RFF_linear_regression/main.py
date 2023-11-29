import models
import utils
import model_utils as m_utils
import numpy as np
import torch
import argparse

PATH_TO_RESULTS = utils.get_path_to('results')
PATH_TO_DATA = utils.get_path_to('data')
PATH_TO_RFFS = 'src/RFF_linear_regression/RFF_data/'

# process arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()
dataset = args.dataset
device = args.device

prepare_data = True

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
N_SAMPLES = 10000  # size of subset of data
N_MAX = 30*(10**3)  # maximum number of fourier features (divided by 2)

if prepare_data:
    print(f'\n\n\nLoading {dataset} data...')
    X_train, y_train, X_test, y_test = utils.process_data(dataset, N_SAMPLES, device)

    np.save(PATH_TO_DATA + 'RFF_y_train.npy', y_train)
    np.save(PATH_TO_DATA + 'RFF_y_test.npy', y_test)
    del y_train
    del y_test

    print('Preparing RFFs...')
    # prepare and save the RFF up front
    v = torch.randn(size=(X_train.shape[1], N_MAX)).to(device) * 0.2 # coefficents of RFF
    RFF_train_full = utils.generate_RFF(v, X_train).to(torch.float16)
    np.save(PATH_TO_DATA + 'RFF_train_full.npy', RFF_train_full)  # save the RFF to memory
    del RFF_train_full
    RFF_test_full = utils.generate_RFF(v, X_test).to(torch.float16)
    np.save(PATH_TO_DATA +'RFF_test_full.npy', RFF_test_full)  # save the RFF to memory
    del RFF_test_full


# load all y values
y_train = np.load(PATH_TO_DATA + 'RFF_y_train.npy')
y_test = np.load(PATH_TO_DATA + 'RFF_y_test.npy')
y_train = torch.from_numpy(y_train).to(torch.float64).to(device)
y_test = torch.from_numpy(y_test).to(torch.float64).to(device)

################ RUN EXPERIMENTS ####################
p_excess_vals = [0, 2500, 5000, 10000, 20000, 50000]
p_ls_vals = [25, 500, 1000, 2500, 5000, 7500, 10000]

results_dict = {}
print('Running first experiment...')
# middle figure increasing parameters for various p_excess values
for p_excess in p_excess_vals:
    print(f'Working on p_excess = {p_excess}')
    N = 10000 + p_excess
    # get subset of data
    RFF_train_N_T, RFF_test_N_T = utils.get_subset_of_RFFs(N, PATH_TO_DATA, device)

    # apply PCA (where appropriate)
    if p_excess > 0:
        RFF_train_N_T, RFF_test_N_T = models.apply_PCA(RFF_train_N_T, RFF_test_N_T, 10000)

    # add bias column
    RFF_train_N_T = utils.add_bias_col(RFF_train_N_T)
    RFF_test_N_T = utils.add_bias_col(RFF_test_N_T)

    eff_p_l2_train = []
    eff_p_l2_test = []
    eff_p_l2_squared_train = []
    eff_p_l2_squared_test = []
    test_errors = []
    train_errors = []
    ind_sqr_test_errors = []

    for p_ls in p_ls_vals:
        print(f'Working on p_ls = {p_ls}')
        # fit model
        LR_list = [models.LinearRegression() for _ in range(10)]

        # fit all models in a one-vs-all fashion
        LR_list = m_utils.fit_all(LR_list, RFF_train_N_T[:, :p_ls], y_train)

        # evaluate error on the test set
        error = m_utils.eval_all(LR_list, RFF_test_N_T[:, :p_ls], y_test)
        test_errors.append(error.item() + 1e-7)
        
        # evaluate individual squared errors for all 10 models in a one-vs-all fashion
        test_errors_ova = m_utils.eval_ind_sqr_all(LR_list, RFF_test_N_T[:, :p_ls], y_test)
        ind_sqr_test_errors.append(test_errors_ova)

        # evaluate error on the train set
        error = m_utils.eval_all(LR_list, RFF_train_N_T[:, :p_ls], y_train)
        train_errors.append(error.item() + 1e-7)
        
        eff_p_l2_squared_train_ova, eff_p_l2_train_ova = m_utils.eff_params_all(LR_list, RFF_train_N_T[:, :p_ls])
        eff_p_l2_squared_test_ova, eff_p_l2_test_ova = m_utils.eff_params_all(LR_list, RFF_test_N_T[:, :p_ls])
        eff_p_l2_squared_train.append(eff_p_l2_squared_train_ova)
        eff_p_l2_squared_test.append(eff_p_l2_squared_test_ova)
        eff_p_l2_train.append(eff_p_l2_train_ova)
        eff_p_l2_test.append(eff_p_l2_test_ova)

        results_dict_run = utils.results_to_dict(eff_p_l2_train, eff_p_l2_test, eff_p_l2_squared_train, 
                                                eff_p_l2_squared_test, test_errors, train_errors, ind_sqr_test_errors,
                                                p_ls_vals, p_excess_vals)
        results_dict[f'p_excess_{p_excess}'] = results_dict_run
        utils.save_results(results_dict, PATH_TO_RESULTS + dataset + '/middle_fig_results.json')


N_vals = [25, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 40000, 60000]

print('\n\n\nRunning third experiment...')
eff_p_l2_train = []
eff_p_l2_test = []
eff_p_l2_squared_train = []
eff_p_l2_squared_test = []
ind_sqr_test_errors = []
test_errors = []
train_errors = []
for i, N in enumerate(N_vals):
    print(f'Working on N = {N}')
    N = int(N)

    RFF_train_N_T, RFF_test_N_T = utils.get_subset_of_RFFs(N, PATH_TO_DATA, device)

    # add bias column
    RFF_train_N_T = utils.add_bias_col(RFF_train_N_T)
    RFF_test_N_T = utils.add_bias_col(RFF_test_N_T)

    # fit models
    LR_list = [models.LinearRegression() for _ in range(10)]
    LR_list = m_utils.fit_all(LR_list, RFF_train_N_T, y_train)

    # evaluate error on the test set
    error = m_utils.eval_all(LR_list, RFF_test_N_T, y_test)
    test_errors.append(error.item() + 1e-7)

    # evaluate individual squared errors for all 10 models in a one-vs-all fashion
    test_errors_ova = m_utils.eval_ind_sqr_all(LR_list, RFF_test_N_T, y_test)
    ind_sqr_test_errors.append(test_errors_ova)

    # calculate effective parameters
    eff_p_l2_squared_train_ova, eff_p_l2_train_ova = m_utils.eff_params_all(LR_list, RFF_train_N_T)
    eff_p_l2_squared_test_ova, eff_p_l2_test_ova = m_utils.eff_params_all(LR_list, RFF_test_N_T)
    eff_p_l2_squared_train.append(eff_p_l2_squared_train_ova)
    eff_p_l2_squared_test.append(eff_p_l2_squared_test_ova)
    eff_p_l2_train.append(eff_p_l2_train_ova)
    eff_p_l2_test.append(eff_p_l2_test_ova)

    # evaluate error on the train set
    error = m_utils.eval_all(LR_list, RFF_train_N_T, y_train)
    train_errors.append(error.item() + 1e-7)
    
    results_dict = utils.results_to_dict(eff_p_l2_train, eff_p_l2_test, eff_p_l2_squared_train, 
                                            eff_p_l2_squared_test, test_errors, train_errors, 
                                            ind_sqr_test_errors, N_vals, [0])
    utils.save_results(results_dict, PATH_TO_RESULTS + dataset + '/left_fig_results.json')
