import pandas as pd
import numpy as np
import dgl
import scipy.sparse as sp
import networkx as nx
import torch


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