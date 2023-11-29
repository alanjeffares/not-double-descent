import utils
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

prepare_data = False

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
p_ls = 10000

s_vals = []
print('Running first experiment...')
# middle figure increasing parameters for various p_excess values
for p_excess in p_excess_vals:
    print(f'Working on p_excess = {p_excess}')
    N = 10000 + p_excess
    # get subset of data
    RFF_train_N_T, RFF_test_N_T = utils.get_subset_of_RFFs(N, PATH_TO_DATA, device)

    # apply SVD
    s = torch.linalg.svdvals(RFF_train_N_T).tolist()
    s_vals.append(s)

# save the list of results locally
np.save(PATH_TO_RESULTS + 'RFF_s_vals.npy', s_vals)



    
