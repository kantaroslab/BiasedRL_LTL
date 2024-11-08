import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


class BiasedDatasetROS(Dataset):
    def __init__(self, data_loc, ws_size, type='Train'):
        # read data to Dataset from pandas and load as tensors
        full_data_ = pd.read_csv(data_loc)
        # full_data = full_data_.sample(frac=1)  # shuffle data first
        full_data = full_data_.iloc[np.random.permutation(len(full_data_))].reset_index(drop=True)
        dlen = len(full_data)
        if type == 'Train':
            self.data = full_data[:int(0.8 * dlen)]
            # self.data = full_data[:1000]
        elif type == 'Test':
            self.data = full_data[int(0.9 * dlen):]

        self.ws_size = ws_size  # size of the gazebo workspace

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        y = self.data.iloc[index, 5]
        """
        data structure: 
        [x0, y0, t0, xf, yf, index, action[0], action[1], x1, y1, t1]
        data normalization
        """
        x_ = np.array(self.data.iloc[index, 0:5])
        x0 = x_[0] / self.ws_size
        y0 = x_[1] / self.ws_size
        t0 = x_[2] / (2 * np.pi)
        xf = x_[3] / self.ws_size
        yf = x_[4] / self.ws_size
        x = torch.FloatTensor(np.array([x0, y0, t0, xf, yf]))
        return x, y


class BiasedDirection(Dataset):
    def __init__(self, data_loc, type='Train'):
        # read data to Dataset from pandas and load as tensors
        self.env = 0
        full_data_ = pd.read_csv(data_loc)
        full_data = full_data_.sample(frac=1)  # shuffle data first
        dlen = len(full_data)
        if type == 'Train':
            self.data = full_data[:int(0.8 * dlen)]
        elif type == 'Test':
            self.data = full_data[int(0.8 * dlen):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        y = self.data.iloc[index, 5]
        """
        data structure: 
        [x0, y0, t0, xf, yf, index, action[0], action[1], x1, y1, t1]
        data normalization
        """
        x_ = np.array(self.data.iloc[index, 0:5])
        x0 = x_[0] / self.env.observation_shape[0]
        y0 = x_[1] / self.env.observation_shape[1]
        t0 = x_[2] / (2 * np.pi)
        xf = x_[3] / self.env.observation_shape[0]
        yf = x_[4] / self.env.observation_shape[1]
        x = torch.FloatTensor(np.array([x0, y0, t0, xf, yf]))
        return x, y


def normalize_data_ros(x_, ws_space):
    x0 = x_[0] / ws_space
    y0 = x_[1] / ws_space
    t0 = x_[2] / (2 * np.pi)
    xf = x_[3] / ws_space
    yf = x_[4] / ws_space
    return np.array([x0, y0, t0, xf, yf])


def normalize_data(x_, env):
    x0 = x_[0] / env.observation_shape[0]
    y0 = x_[1] / env.observation_shape[1]
    t0 = x_[2] / (2 * np.pi)
    xf = x_[3] / env.observation_shape[0]
    yf = x_[4] / env.observation_shape[1]
    return np.array([x0, y0, t0, xf, yf])


class Network(nn.Module):
    def __init__(self, net_dims, activation=nn.ReLU):
        super(Network, self).__init__()
        layers = []
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))
            if i != len(net_dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        x = self.softmax(x)
        return x

class Network_(nn.Module):
    def __init__(self, net_dims, activation=nn.ReLU):
        super(Network, self).__init__()
        layers = []
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))
            if i != len(net_dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_data_loader_ros(BATCH_SIZE, data_location, ws_size):
    train_dataset = BiasedDatasetROS(data_location, ws_size, type='Train')
    test_dataset = BiasedDatasetROS(data_location, ws_size, type='Test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                                              drop_last=False)
    print("Length of train_set: {} | test_set: {}".format(len(train_dataset), len(test_dataset)))
    return train_loader, test_loader


def create_data_loader(BATCH_SIZE, data_location):
    train_dataset = BiasedDirection(data_location, type='Train')
    test_dataset = BiasedDirection(data_location, type='Test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                              drop_last=False)
    print("Length of train_set: {} | test_set: {}".format(len(train_dataset), len(test_dataset)))
    return train_loader, test_loader


def check_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)
