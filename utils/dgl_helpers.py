import pandas as pd
import numpy as np
import dgl
import scipy.sparse as sp
import networkx as nx
import torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class TemporalConvLayer(nn.Module):
    ''' Temporal convolution layer.

    arguments
    ---------
    c_in : int
        The number of input channels (features)
    c_out : int
        The number of output channels (features)
    dia : int
        The dilation size
    '''
    def __init__(self, c_in, c_out, dia = 1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        print(c_in, c_out)
        self.conv = nn.Conv2d(c_in, c_out, (2, 1), 1, dilation = dia, padding = (0,0))


    def forward(self, x):
        return torch.relu(self.conv(x))

class SpatioConvLayer(nn.Module):
    def __init__(self, c, Lk): # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        self.gc = GraphConv(c, c, activation=F.relu)

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x.transpose(0, 3)
        x = x.transpose(1, 3)
        output = self.gc(self.g, x)
        output = output.transpose(1, 3)
        output = output.transpose(0, 3)
        return torch.relu(output)

class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = FullyConvLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

blocks = [1, 16, 32, 64, 32, 128]

class STGCN_WAVE(nn.Module):
    def __init__(self, c, T, n, Lk, device=torch.device("cuda"), control_str='TNTSTNTST'):
        super(STGCN_WAVE, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            print("DONEEEE", i)
            i_layer = control_str[i]
            if i_layer == 'T': # Temporal Layer
                self.layers.append(TemporalConvLayer(c[cnt], c[cnt + 1], dia = 2**diapower))
                diapower += 1
                cnt += 1
            if i_layer == 'S': # Spatio Layer
                self.layers.append(SpatioConvLayer(c[cnt], Lk))
            if i_layer == 'N': # Norm Layer
                self.layers.append(nn.LayerNorm([n,c[cnt]]))
        self.output = OutputLayer(c[cnt], T + 1 - 2**(diapower), n)
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x):
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == 'N':
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.layers[i](x)
        return self.output(x)


save_path = "stgcnwavemodel.pt"

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse, wmape = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)

            mae += d.tolist()
            # np.sum(d) / np.sum(np.abs(y))
            wmape += np.abs(y).tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        WMAPE = np.array(mae).sum() / np.array(wmape).sum()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, WMAPE, RMSE





def get_data(start_date="2020-10-31", end_date="2021-01-01"):
    data = pd.read_parquet("data/wind/2019-01-24_outlier_removed.parquet")
    data = data[~data["rt_plant_id"].isin([2397, 2420, 2538])]
    # bu üç plant sonradan dahil oluyor
    data = data[~data["rt_plant_id"].isin([1470])]
    # bu plantin correlationu düşük diğerlerine göre
    assert data.rt_plant_id.nunique() == 97
    plant_mapping = {k:v for k,v in zip(np.sort(data.rt_plant_id.astype(int).unique()), range(98))}
    # rev_plant_mapping = {v:k for k,v in plant_mapping.items()}
    data.rt_plant_id = data.rt_plant_id.map(plant_mapping)
    data = data[(data["forecast_dt"] > start_date) & (data["forecast_dt"] < end_date)]
    print(data.forecast_dt.nunique(), data.forecast_dt.nunique() / 24)
    weather_cols = [col for col in data.columns if col.startswith(("VGRD", "UGRD"))]
    data_pivot = pd.pivot_table(
        data[["rt_plant_id", "forecast_dt", "production", *weather_cols]],
        index="forecast_dt",
        columns="rt_plant_id",
        )
    return data_pivot

def split(data, train_ratio, val_ratio):
    # indices might be useless
    # might return dataframes directly in the future
    train_end = int(len(data) * train_ratio)
    val_end = int(len(data) * (train_ratio + val_ratio))
    train_indices = data.index[:train_end]
    val_indices = data.index[train_end:val_end]
    test_indices = data.index[val_end:]
    return train_indices, val_indices, test_indices

def generate_adjacency_matrix(df, threshold=0.5):
    A = abs(df.corr())
    A = A[A > threshold].fillna(0)
    return A

def create_graph(adj):
    sp_mx = sp.coo_matrix(adj)
    G = dgl.from_scipy(sp_mx, eweight_name="weight")
    print(len(G.all_edges()[0]))
    print(len(G.edata["weight"]))
    G = dgl.remove_self_loop(G)
    # isolates = list(nx.isolates(G.to_networkx()))
    isolates = [i for i in range(len(adj)) if i not in G.all_edges()[0]]
    assert len(isolates) == 0, "there are isolated nodes: {}".format(isolates)
    G = dgl.add_self_loop(G)
    # print(G.edata["weight"])
    return G


def data_transform(data, n_window, n_ahead, device=torch.device("cpu")):
    """
    :param data: rows for time indices, columns for locations
        currently does not support features
    :param n_window: history length
    :param n_ahead: next time index to predict

    X : matrix of node features, X.shape = (B, N, F, T)
        B : batch size
        N : number of nodes
        F : number of features
        T : number of time steps
    """
    n_obs, n_loc = data["production"].shape
    num = n_obs - n_window - n_ahead + 1
    feature_cols = data.columns.get_level_values(0).unique().tolist()
    n_features = len(feature_cols)

    x = np.zeros([num, n_features, n_window, n_loc]) # features might end up being in second dimension
    y = np.zeros([num, n_loc])

    for i, col in enumerate(feature_cols):
        for start in range(num):
            x[start, i, :, :] = data[col][start: start + n_window].values.reshape(1, n_window, n_loc)
            if col == "production":
                y[start, :] = data[col].values[start + n_window + n_ahead - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def prepare_torch_data(x, y, train_indices, val_indices, n_window, batch_size):
    x_train = x[:len(train_indices) - n_window, :, :, :]
    x_val = x[len(train_indices) - n_window: len(train_indices) - n_window + len(val_indices), :, :, :]
    # x_test = x[len(train_indices) - n_window + len(val_indices):, :, :, :]

    y_train = y[:len(train_indices) - n_window, :]
    y_val = y[len(train_indices) - n_window: len(train_indices) - n_window + len(val_indices), :]
    # y_test = y[len(train_indices) - n_window + len(val_indices):, :]

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size)
    return train_iter, val_iter

n_window = 24
n_loc = 97

def convert_x_tensor_to_df(x, start_index, feature_index):
    data_ = x[start_index][feature_index].cpu().numpy().reshape(n_window, n_loc)
    index_ = data.index[start_index:start_index+n_window]
    columns_ = pd.MultiIndex.from_product([[data.columns.get_level_values(0).unique()[feature_index]], range(n_loc)])
    return pd.DataFrame(data_, index=index_, columns=columns_)

def convert_y_tensor_to_df(y, start_index):
    data_ = y[start_index].cpu().numpy().reshape(1, n_loc)
    index_ = [data.index[start_index+n_window]]
    columns_ = pd.MultiIndex.from_product([["production"], range(n_loc)])
    return pd.DataFrame(data_, index=index_, columns=columns_)