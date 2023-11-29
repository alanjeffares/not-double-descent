import torchvision
import torch
import json

from sklearn.preprocessing import OneHotEncoder

N_SAMPLES = 10000  # size of subset of data


def get_path_to(folder):
    with open('src/config.json') as f:
        config = json.load(f)
    return config[f'{folder}_folder']


# function that takes a torch dataset and returns 10000 inputs and labels which are balanced for each class
def balance_data_set(dataset, num_samples=10000):
    """Subsample the data set to have balanced classes."""
    # get the indices of each class
    indices = [torch.where(dataset.targets == i)[0] for i in range(10)]
    # get the first 1000 indices of each class
    num_per_class = int(num_samples / 10)
    indices = [i[:num_per_class] for i in indices]
    # get the first 1000 indices of each class
    indices = torch.cat(indices)
    # get the inputs and labels
    inputs = dataset.data[indices]
    labels = dataset.targets[indices]
    return inputs, labels


def get_raw_data(dataset: str, split: str):
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
        X_test = testset.data
        y_test = testset.targets

    X_train = X_train / 255
    X_train = X_train.reshape(num_samples, -1).to(device)
    y_train = y_train.to(device)

    X_test = X_test.reshape(X_test.shape[0], -1) / 255
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    y_train = y_train.to(torch.float64)
    y_test = y_test.to(torch.float64)

    return X_train, y_train, X_test, y_test


def get_data(dataset, n_samples=N_SAMPLES, pred_idx='all', center: bool = False):

    X_train, y_train, X_test, y_test = process_data(dataset, n_samples, 'cpu')
    X_train, y_train, X_test, y_test = X_train.numpy(), y_train.numpy(), X_test.numpy(), \
                                       y_test.numpy()

    ohe = OneHotEncoder(sparse_output=False)
    y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
    y_test_ohe = ohe.transform(y_test.reshape(-1, 1))

    if center:
        y_train_ohe = y_train_ohe - .5
        y_test_ohe = y_test_ohe - .5

    if pred_idx == 'all':
        return X_train, y_train_ohe, X_test, y_test_ohe
    else:
        return X_train, y_train_ohe[:, pred_idx], X_test, y_test_ohe[:, pred_idx]
