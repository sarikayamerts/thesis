import pandas as pd
import numpy as np
import os

if not os.environ.get("FOLDER_PATH"):
    import subprocess; FOLDER_PATH = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
else:
    FOLDER_PATH = os.environ["FOLDER_PATH"]

def read_data(generate_speed_angle=False,
              add_lagged=False,
              number_of_plants=94):
    try:
        df = pd.read_parquet(f"{FOLDER_PATH}/data/processed/outlier_removed.parquet")
    except:
        df = pd.read_parquet("https://storage.googleapis.com/wind_power_forecast/data/processed/outlier_removed.parquet")
    if add_lagged:
        df["production_48_lagged"] = df.groupby("rt_plant_id").production.shift(48)
        df = df.dropna()
    weather_cols = [col for col in df.columns if col.startswith(("UGRD", "VGRD"))]

    assert 1 <= number_of_plants <= 94
    if number_of_plants < 94:
        plants = df.groupby("rt_plant_id").production.sum().sort_values(ascending=False).index[:number_of_plants]
        print("Selected plants:\n", plants.to_list())
        df = df[df["rt_plant_id"].isin(plants)]

    if add_lagged:
        cols = ["rt_plant_id", "production", "production_48_lagged", *weather_cols]
    else:
        cols = ["rt_plant_id", "production", *weather_cols]

    df = df.set_index("forecast_dt")[cols]

    if generate_speed_angle:
        for box in ["SW", "NW", "NE", "SE"]:
            df[f"speed_{box}"] = np.sqrt(np.square(df[f"UGRD_80.m.above.ground.{box}"]) + np.square(df[f"VGRD_80.m.above.ground.{box}"]))
            df[f"angle_{box}"] = np.arctan(df[f"UGRD_80.m.above.ground.{box}"] / df[f"VGRD_80.m.above.ground.{box}"])
    return df

def _expand_plant_dimension(df):
    PLANTS = sorted(df.rt_plant_id.unique())
    n_loc = len(PLANTS)
    n_time = df.index.nunique()
    cols = [col for col in df.columns if col != "rt_plant_id"]
    n_cols = len(cols)

    df_np = np.zeros((n_time, n_loc, n_cols))
    for i, plant_id in enumerate(PLANTS):
        df_np[:, i, :] = df[df.rt_plant_id == plant_id][cols].values
    return df_np

def scale_data(train_df, valid_df, test_df, scaler=None, expand=False):
    PLANTS = sorted(train_df.rt_plant_id.unique())

    if scaler is not None:
        import pickle
        assert scaler in ["minmax", "standart"]

        scalers = {}
        lower_bound = 1e-8

        if scaler == "minmax":
            from sklearn.preprocessing import MinMaxScaler as scaler_
        else:
            from sklearn.preprocessing import StandartScaler as scaler_

    if expand:
        train_df = _expand_plant_dimension(train_df)
        valid_df = _expand_plant_dimension(valid_df)
        test_df = _expand_plant_dimension(test_df)
        for i, plant in enumerate(PLANTS):
            scalers[plant] = scaler_()
            train_df[:, i, :] = scalers[plant].fit_transform(train_df[:, i, :]).clip(min=lower_bound, max=1-lower_bound)
            valid_df[:, i, :] = scalers[plant].transform(valid_df[:, i, :]).clip(min=lower_bound, max=1-lower_bound)
            test_df[:, i, :] = scalers[plant].transform(test_df[:, i, :]).clip(min=lower_bound, max=1-lower_bound)
    else:
        for i, plant in enumerate(PLANTS):
            scalers[plant] = scaler_()
            train_df.loc[train_df["rt_plant_id"] == plant, train_df.columns != "rt_plant_id"] = scalers[plant].fit_transform(
                train_df.loc[train_df["rt_plant_id"] == plant, train_df.columns != "rt_plant_id"]).clip(min=lower_bound, max=1-lower_bound)
            valid_df.loc[valid_df["rt_plant_id"] == plant, valid_df.columns != "rt_plant_id"] = scalers[plant].transform(
                valid_df.loc[valid_df["rt_plant_id"] == plant, valid_df.columns != "rt_plant_id"]).clip(min=lower_bound, max=1-lower_bound)
            test_df.loc[test_df["rt_plant_id"] == plant, test_df.columns != "rt_plant_id"] = scalers[plant].transform(
                test_df.loc[test_df["rt_plant_id"] == plant, test_df.columns != "rt_plant_id"]).clip(min=lower_bound, max=1-lower_bound)

    if scaler is not None:
        with open(f'{FOLDER_PATH}/artifacts/scalers.pickle', 'wb') as handle:
            pickle.dump(scalers, handle)

    return train_df, valid_df, test_df

def split_data(df, train_ratio=0.8, valid_ratio=0.1):
    time_indices = sorted(df.index.unique())

    train_indices = time_indices[:int(len(time_indices) * train_ratio)]
    valid_indices = time_indices[int(len(time_indices) * train_ratio):int(len(time_indices) * (train_ratio + valid_ratio))]
    test_indices = time_indices[int(len(time_indices) * (train_ratio + valid_ratio)):]

    print("Train start and end dates:\t", train_indices[0], "\t", train_indices[-1])
    try:
        print("Validation start and end dates:\t", valid_indices[0], "\t", valid_indices[-1])
    except:
        pass
    print("Test start and end dates:\t", test_indices[0], "\t", test_indices[-1])

    train_df = df.loc[train_indices, :]
    valid_df = df.loc[valid_indices, :]
    test_df = df.loc[test_indices, :]

    return train_df, valid_df, test_df

def plot_metrics(performance, val_performance, metric_index=-1):
    import matplotlib.pyplot as plt
    x = np.arange(len(performance))
    width = 0.3
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in performance.values()]

    # plt.ylabel('mean_absolute_error [production, normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
    _ = plt.legend(loc="upper right")

def calculate_wmape(preds, actuals):
    return np.sum(np.abs(preds-actuals)) / np.sum(np.abs(actuals))
