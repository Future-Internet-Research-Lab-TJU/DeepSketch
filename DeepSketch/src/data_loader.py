import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

def training_data(path, batch_size):
    train_data = pd.read_csv(path)
    train_X = train_data.iloc[:, [1, 3, 5]].values
    train_y = train_data.iloc[:, 6].values
    train_X_min = train_X.min()
    train_X_max = train_X.max()
    train_X = (train_X - train_X_min) / (train_X_max - train_X_min)
    train_y_min = train_y.min()
    train_y_max = train_y.max()
    train_y = (train_y - train_y_min) / (train_y_max - train_y_min)
    train_X_t = torch.from_numpy(train_X.astype(np.float32)).unsqueeze(-1)
    train_y_t = torch.from_numpy(train_y.astype(np.float32)).unsqueeze(-1)
    train_data_s  = Data.TensorDataset(train_X_t, train_y_t)
    train_data_loader = Data.DataLoader(
        dataset=train_data_s,
        batch_size=batch_size,
        shuffle=False
    )
    return train_data_loader, train_X_min, train_X_max, train_y_min, train_y_max

def testing_data(path, batch_size):
    test_data = pd.read_csv(path)
    test_X = test_data.iloc[:, [1, 3, 5]].values
    test_y = test_data.iloc[:, 6].values
    test_X_t = torch.from_numpy(test_X.astype(np.float32)).unsqueeze(-1)
    test_y_t = torch.from_numpy(test_y.astype(np.float32)).unsqueeze(-1)
    packet_num = torch.sum(test_y_t, dim=0).item()
    test_data_s  = Data.TensorDataset(test_X_t, test_y_t)
    test_data_loader = Data.DataLoader(
        dataset=test_data_s,
        batch_size=batch_size,
        shuffle=True
    )
    return packet_num, test_data_loader