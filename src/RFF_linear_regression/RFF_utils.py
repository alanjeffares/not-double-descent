import torchvision
import torch
import numpy as np
import json

def interleave_tensor(tensor):
    length = tensor.shape[0]
    indices = torch.zeros(length, dtype=torch.long)
    indices[0:length:2] = torch.arange(0, length // 2)
    indices[1:length:2] = torch.arange(length // 2, length)
    return tensor[indices], indices

def process_MNIST(num_samples=10000, device='cpu'):
    # download and process MNIST
    trainset = torchvision.datasets.MNIST(
        root='../../../../data/aj659.data/', train=True, download=True)

    testset = torchvision.datasets.MNIST(
        root='../../../../data/aj659.data/', train=False, download=True)

    X_train = trainset.data[:num_samples]/255
    X_train = X_train.reshape(num_samples,-1).to(device)
    y_train = trainset.targets[:num_samples].to(device)

    X_test = testset.data.reshape(10000,-1)/255
    X_test = X_test.to(device)
    y_test = testset.targets.to(device)

    # balanced train set
    idx_0 = trainset.targets == 0
    idx_1 = trainset.targets == 1
    X_train_0 = trainset.data[idx_0][:int(num_samples/2)]/255
    X_train_1 = trainset.data[idx_1][:int(num_samples/2)]/255
    X_train = torch.cat([X_train_0, X_train_1], 0)  # full X_train

    y_train_0 = trainset.targets[idx_0][:int(num_samples/2)]
    y_train_1 = trainset.targets[idx_1][:int(num_samples/2)]
    y_train = torch.cat([y_train_0, y_train_1], 0)  # full y_train

    # balanced test set
    idx_0 = testset.targets == 0
    idx_1 = testset.targets == 1
    X_test_0 = testset.data[idx_0][:1000]/255
    X_test_1 = testset.data[idx_1][:1000]/255
    X_test = torch.cat([X_test_0, X_test_1], 0)  # full X_test

    y_test_0 = testset.targets[idx_0][:1000]
    y_test_1 = testset.targets[idx_1][:1000]
    y_test = torch.cat([y_test_0, y_test_1], 0)  # full y_test

    # shuffle the training data such that all subsets are balanced too
    y_train, idx = interleave_tensor(y_train)
    X_train = X_train[idx]

    # reshape
    X_train = X_train.reshape(num_samples,-1).to(device)
    y_train = y_train.to(device)
    X_test = X_test.reshape(1980,-1).to(device)
    y_test = y_test.to(device)

    y_train = (y_train == 0)*1. - 0.5
    y_test = (y_test == 0)*1. - 0.5
    y_train = y_train.to(torch.float64)
    y_test = y_test.to(torch.float64)
    
    return X_train, y_train, X_test, y_test

def generate_RFF(v, X):
    num_FF = v.shape[1]
    num_samples = X.shape[0]
    RFF_train = torch.einsum("ij, jk -> ik", X, v)
    # converting to polar form -> e^(i \theta) = cos(\theta) + i sin(\theta)
    RFF_train_real = torch.cos(RFF_train)  # real part of phi
    RFF_train_img = torch.sin(RFF_train)  # imaginary part of phi
    # combine real and imaginary into a single array of dim (num_samples, 2 * N_max)
    RFF_train_full = torch.empty((num_samples, 2 * num_FF), dtype=RFF_train_real.dtype)
    RFF_train_full[:, 0::2] = RFF_train_real
    RFF_train_full[:, 1::2] = RFF_train_img
    return RFF_train_full


def add_bias_col(data):
    num_samples = data.shape[0]
    bias_term = torch.ones(num_samples, 1)  # add bias term
    return torch.hstack([bias_term, data])


class LinearRegression:
    def _init_(self):
        pass
    def fit(self, X, y):
        # fully determined case (standard least squares)
        if X.shape[0] >= X.shape[1]:
            self.ytobeta = torch.linalg.solve(X.t() @ X, X.t())
            self.beta = self.ytobeta @ y
        # underdetermined case with min norm solution (see: https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf)
        elif X.shape[0] < X.shape[1]:
            # self.ytobeta = X.t() @ torch.inverse(X @ X.t())
            # self.ytobeta = torch.linalg.solve(X @ X.t(), X).t()
            self.ytobeta = torch.linalg.pinv(X)  # this is the most numerically stable method
            self.beta = self.ytobeta @ y
        self.X = X; self.y = y  # store data

    def eff_p_l2(self, X_eval):
        S = X_eval @ self.ytobeta
        return torch.mean(torch.linalg.vector_norm(S, dim=1))
    
    def eff_p_l2_squared(self, X_eval):
        S = X_eval @ self.ytobeta
        return torch.mean(torch.linalg.vector_norm(S, dim=1)**2)

    def predict(self, X):
        return X @ self.beta

    def weight_norm(self):
        return torch.linalg.vector_norm(self.beta, ord=2)


class PCA:
    def __init__(self):
        self.f_means = None
        self.f_stds = None

    def fit(self, X, n_components):
        X_stand = self._standardize(X)
        # E, V = torch.linalg.eigh(X_stand.t() @ X_stand)
        # key = torch.argsort(E).flip(0)[:n_components]
        # T = X_stand @ V[:, :n_components]
        # self.V = V[:, key]

        _, _, self.V = torch.pca_lowrank(X_stand, q=n_components, center=False, niter=10)

    def transform(self, X):
        X_stand = self._standardize(X)
        return X_stand @ self.V

    def fit_transform(self, X, n_components):
        self.fit(X, n_components)
        return self.transform(X)

    def _standardize(self, X):
        if self.f_means is None:
            self.f_means = torch.mean(X, 0)
        if self.f_stds is None:
            self.f_stds = torch.std(X, 0)
        X = X - self.f_means
        X = X / self.f_stds
        return X

def apply_PCA(X_train, X_test, n_components):
    pca = PCA()
    X_train = pca.fit_transform(X_train, n_components)
    X_test = pca.transform(X_test)
    return X_train, X_test

class Standardize:
    def __init__(self):
        self.f_means = None
        self.f_stds = None

    def fit_transform(self, X):
        if self.f_means is None:
            self.f_means = torch.mean(X, 0)
        if self.f_stds is None:
            self.f_stds = torch.std(X, 0)
        X = X - self.f_means
        X = X / self.f_stds
        return X
    
# function that stores multiple lists of results into a dictionary
def results_to_dict(eff_p_l2_train, eff_p_l2_test, eff_p_l2_squared_train, 
                    eff_p_l2_squared_test, test_errors, train_errors, p_x_axis, 
                    p_y_axis):
    results = {}
    results['eff_p_l2_train'] = eff_p_l2_train
    results['eff_p_l2_test'] = eff_p_l2_test
    results['eff_p_l2_squared_train'] = eff_p_l2_squared_train
    results['eff_p_l2_squared_test'] = eff_p_l2_squared_test
    results['test_errors'] = test_errors
    results['train_errors'] = train_errors
    results['p_x_axis'] = p_x_axis
    results['p_y_axis'] = p_y_axis
    return results

# function that saves a dict of results to a json file
def save_results(results, filename): 
    with open(filename, 'w') as f:
        json.dump(results, f)

def get_subset_of_RFFs(N, path, device):
    # get subset of data
    RFF_train_full = np.load(path + 'RFF_train_full.npy', mmap_mode='r')
    RFF_train_N = RFF_train_full[:, :N]
    RFF_train_N_T = torch.from_numpy(RFF_train_N).to(torch.float64).to(device)
    del RFF_train_full
    del RFF_train_N
    torch.cuda.empty_cache()

    RFF_test_full = np.load(path + 'RFF_test_full.npy', mmap_mode='r')
    RFF_test_N = RFF_test_full[:, :N]
    RFF_test_N_T = torch.from_numpy(RFF_test_N).to(torch.float64).to(device)
    del RFF_test_full
    del RFF_test_N
    torch.cuda.empty_cache()
    return RFF_train_N_T, RFF_test_N_T