import numpy as np


def split_data(x, p_train, p_valid):
    train_size = int(p_train * x.shape[0])
    valid_size = int(p_valid * x.shape[0])
    index = np.random.permutation(x.shape[0])
    idx_train = index[:train_size]
    idx_valid = index[index[train_size:(train_size + valid_size)]]
    train = x[idx_train, :]
    valid = x[idx_valid, :]
    if train_size + valid_size != 1:
        idx_test = index[(train_size + valid_size):]
        test = x[idx_test, :]
    else:
        test = np.NaN
    return train, valid, test, idx_train
