import torchvision
import torch
import numpy as np
import json

def get_path_to(folder):
    with open('src/config.json') as f:
        config = json.load(f)
    return config[f'{folder}_folder']

class Dataset:
    def __init__(self):
        self.data = None
        self.targets = None

def interleave_tensor(tensor):
    length = tensor.shape[0]
    indices = torch.zeros(length, dtype=torch.long)
    indices[0:length:2] = torch.arange(0, length // 2)
    indices[1:length:2] = torch.arange(length // 2, length)
    return tensor[indices], indices

# function that takes a torch dataset and returns 10000 inputs and labels which are balanced for each class
def balance_data_set(dataset, num_samples=10000): 
    """Subsample the data set to have balanced classes."""
    # get the indices of each class
    indices = [torch.where(dataset.targets == i)[0] for i in range(10)]
    # get the first 1000 indices of each class
    num_per_class = int(num_samples/10)
    indices = [i[:num_per_class] for i in indices]
    # get the first 1000 indices of each class
    indices = torch.cat(indices)
    # get the inputs and labels
    inputs = dataset.data[indices]
    labels = dataset.targets[indices]
    return inputs, labels

def get_raw_data(dataset:str, split:str):
    """Logic for loading raw MNIST, CIFAR-10 and SVHN datasets"""
    if dataset == 'MNIST':
        if split == 'train':
            return torchvision.datasets.MNIST(
                root=get_path_to('data'), train=True, download=True)
        else:
            return torchvision.datasets.MNIST(
                root=get_path_to('data'), train=False, download=True)
        
    elif dataset == 'CIFAR-10':  # also need to apply greyscale
        if split == 'train':
            # get CIFAR-10 training set and apply greyscale
            trainset = torchvision.datasets.CIFAR10(
                root=get_path_to('data'), train=True, download=True) 
            trainset.data = torch.from_numpy(trainset.data)
            trainset.targets = torch.tensor(trainset.targets)
            trainset.data = torch.mean(trainset.data.float(), axis=-1)  # apply greyscale
            return trainset
        else:
            # get CIFAR-10 test set and apply greyscale
            testset = torchvision.datasets.CIFAR10(
                root=get_path_to('data'), train=False, download=True)
            testset.data = torch.from_numpy(testset.data)
            testset.targets = torch.tensor(testset.targets)        
            testset.data = torch.mean(testset.data.float(), axis=-1)  # apply greyscale
            return testset
        
    elif dataset == 'SVHN':
        # get SVHN dataset and apply greyscale
        dataset = torchvision.datasets.SVHN(
            root=get_path_to('data'), split=split, download=True)
        dataset.data = torch.tensor(dataset.data)
        dataset.targets = torch.tensor(dataset.labels)        
        dataset.data = torch.mean(dataset.data.float(), axis=1)  # apply greyscale
        return dataset


def process_data(dataset, num_samples=10000, device='cpu'):
    trainset = get_raw_data(dataset, 'train')
    testset = get_raw_data(dataset, 'test')
    
    X_train, y_train = balance_data_set(trainset, num_samples)
    if testset.data.shape[0] > 10000:
        X_test, y_test = balance_data_set(testset, 10000)
    else:
        X_test, y_test = testset.data, testset.targets

    X_train = X_train/255
    X_train = X_train.reshape(num_samples,-1).to(device)
    y_train = y_train.to(device)

    X_test = X_test.reshape(X_test.shape[0],-1)/255
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    y_train = y_train.to(torch.float64)
    y_test = y_test.to(torch.float64)
    
    return X_train, y_train, X_test, y_test

    
def process_MNIST(num_samples=10000, device='cpu'):
    # download and process MNIST
    trainset = torchvision.datasets.MNIST(
        root=get_path_to('data'), train=True, download=True)

    testset = torchvision.datasets.MNIST(
        root=get_path_to('data'), train=False, download=True)
    
    X_train, y_train = balance_data_set(trainset, num_samples)

    X_train = X_train/255
    X_train = X_train.reshape(num_samples,-1).to(device)
    y_train = y_train.to(device)

    X_test = testset.data.reshape(10000,-1)/255
    X_test = X_test.to(device)
    y_test = testset.targets.to(device)

    # # y_train = (y_train == 0)*1. - 0.5
    # # y_test = (y_test == 0)*1. - 0.5
    y_train = y_train.to(torch.float64)
    y_test = y_test.to(torch.float64)
    
    return X_train, y_train, X_test, y_test


def ova_labels(target: int, targets: torch.tensor):
    """
    One-vs-all labels for a given target class taking values of 
    +/- 0.5 as float64 type.
    """
    # return ((targets == target)*1. - 0.5).to(torch.float64)
    return ((targets == target)*1.).to(torch.float64)


def process_0_vs_1_MNIST(num_samples=10000, device='cpu'):
    # download and process MNIST
    trainset = torchvision.datasets.MNIST(
        root=get_path_to('data'), train=True, download=True)

    testset = torchvision.datasets.MNIST(
        root=get_path_to('data'), train=False, download=True)

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
                    eff_p_l2_squared_test, test_errors, train_errors, ind_sqr_test_errors,
                    p_x_axis, p_y_axis):
    results = {}
    results['eff_p_l2_train'] = eff_p_l2_train
    results['eff_p_l2_test'] = eff_p_l2_test
    results['eff_p_l2_squared_train'] = eff_p_l2_squared_train
    results['eff_p_l2_squared_test'] = eff_p_l2_squared_test
    results['test_errors'] = test_errors
    results['train_errors'] = train_errors
    results['ind_sqr_test_errors'] = ind_sqr_test_errors
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