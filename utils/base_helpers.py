import pandas as pd
import numpy as np
import os

if not os.environ.get("FOLDER_PATH"):
    import subprocess; FOLDER_PATH = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
else:
    FOLDER_PATH = os.environ["FOLDER_PATH"]

class DataReader:
    def __init__(self, number_of_plants) -> None:
        self.number_of_plants = number_of_plants
        self.raw_df = None
        self.df = None

        self.id_cols = ["rt_plant_id", "production"]
        self.weather_cols = None

        self.train_df = None
        self.valid_df = None
        self.test_df = None

        self.train_indices = None
        self.valid_indices = None
        self.test_indices = None

        self.corr_ordered_plants = None

        # main function after initialization
        # self.process()

    @property
    def plants(self):
        return sorted(self.raw_df.groupby("rt_plant_id").production.sum().sort_values(ascending=False).index[:self.number_of_plants].to_list())

    @property
    def time_indices(self):
        return sorted(self.df.index.unique())

    def process(self, add_speed=True, add_lagged=True,
                train_ratio=0.8, valid_ratio=0.1, test_ratio=None,
                expand=True, scaler="minmax", corr_order=False):
        self.read(add_speed, add_lagged)
        self.split(train_ratio, valid_ratio, test_ratio)
        self.expand(expand, corr_order=corr_order)
        self.scale(scaler)

    def read(self, add_speed, add_lagged):
        self._read_parquet()
        self._add_lagged(add_lagged)
        self.df = self.df.set_index("forecast_dt")[[*self.id_cols, *self.weather_cols]]
        self._add_speed(add_speed)

    def split(self, train_ratio, valid_ratio, test_ratio, verbose=1):
        time_indices = self.time_indices
        if test_ratio is None:
             test_ratio = 1 - train_ratio - valid_ratio
        self.train_indices = time_indices[:int(len(time_indices) * train_ratio)]
        self.valid_indices = time_indices[int(len(time_indices) * train_ratio):int(len(time_indices) * (train_ratio + valid_ratio))]
        self.test_indices = time_indices[int(len(time_indices) * (train_ratio + valid_ratio)): int(len(time_indices) * (train_ratio + valid_ratio + test_ratio))]
        if verbose:
            print("Train start and end dates:\t", self.train_indices[0], "\t", self.train_indices[-1])
            try:
                print("Validation start and end dates:\t", self.valid_indices[0], "\t", self.valid_indices[-1])
            except:
                pass
            print("Test start and end dates:\t", self.test_indices[0], "\t", self.test_indices[-1])

        self.train_df = self.df.loc[self.train_indices, :]
        self.valid_df = self.df.loc[self.valid_indices, :]
        self.test_df = self.df.loc[self.test_indices, :]

    def expand(self, expand, corr_order=None, initial_start=7):
        if expand:
            self.train_df = self._expand_plant_dimension(self.train_df)
            self.valid_df = self._expand_plant_dimension(self.valid_df)
            self.test_df = self._expand_plant_dimension(self.test_df)
        if corr_order:
            self.correlation_ordering(initial_start=initial_start)


    def scale(self, scaler):
        if scaler is None:
            return
        import pickle
        assert scaler in ["minmax", "standart"]

        scalers = {}
        lower_bound = 1e-8

        if scaler == "minmax":
            from sklearn.preprocessing import MinMaxScaler as scaler_
        else:
            from sklearn.preprocessing import StandartScaler as scaler_

        for i, plant in enumerate(self.plants):
            scalers[plant] = scaler_()
            if isinstance(self.train_df, pd.DataFrame):
                cols = [col for col in self.train_df.columns if col != "rt_plant_id"]
                self.train_df.loc[self.train_df["rt_plant_id"] == plant, cols] = scalers[plant].fit_transform(
                    self.train_df.loc[self.train_df["rt_plant_id"] == plant, cols]).clip(min=lower_bound, max=1-lower_bound)
                self.valid_df.loc[self.valid_df["rt_plant_id"] == plant, cols] = scalers[plant].transform(
                    self.valid_df.loc[self.valid_df["rt_plant_id"] == plant, cols]).clip(min=lower_bound, max=1-lower_bound)
                self.test_df.loc[self.test_df["rt_plant_id"] == plant, cols] = scalers[plant].transform(
                    self.test_df.loc[self.test_df["rt_plant_id"] == plant, cols]).clip(min=lower_bound, max=1-lower_bound)
            else:
                self.train_df[:, i, :] = scalers[plant].fit_transform(self.train_df[:, i, :]).clip(min=lower_bound, max=1-lower_bound)
                self.valid_df[:, i, :] = scalers[plant].transform(self.valid_df[:, i, :]).clip(min=lower_bound, max=1-lower_bound)
                self.test_df[:, i, :] = scalers[plant].transform(self.test_df[:, i, :]).clip(min=lower_bound, max=1-lower_bound)

        with open(f'{FOLDER_PATH}/artifacts/scalers.pickle', 'wb') as handle:
            pickle.dump(scalers, handle)

    def correlation_ordering(self, initial_start=7):
        plants = self.plants.copy()
        corr = pd.pivot_table(
            self.raw_df[["rt_plant_id", "production", "forecast_dt"]],
            index="forecast_dt", columns="rt_plant_id",
            values="production").corr()
        ordered_plant_ids, ordered_plants = [], []
        to_append = initial_start
        ordered_plant_ids.append(plants[to_append])
        ordered_plants.append(to_append)

        for _ in range(len(plants)-1):
            corr_series = corr.iloc[to_append].drop(labels=ordered_plant_ids)
            to_append = plants.index(corr_series.idxmax())
            ordered_plant_ids.append(plants[to_append])
            ordered_plants.append(to_append)

        self.corr_ordered_plants = [plants[i] for i in ordered_plants]

        self.train_df = self.train_df[:, ordered_plants, :]
        self.valid_df = self.valid_df[:, ordered_plants, :]
        self.test_df = self.test_df[:, ordered_plants, :]

    def _expand_plant_dimension(self, df):
        n_loc = len(self.plants)
        n_time = df.index.nunique()
        cols = [col for col in self.df.columns if col != "rt_plant_id"]
        n_cols = len(cols)

        df_np = np.zeros((n_time, n_loc, n_cols))
        for i, plant_id in enumerate(self.plants):
            df_np[:, i, :] = df[df.rt_plant_id == plant_id][cols].values
        return df_np

    def _read_parquet(self):
        try:
            df = pd.read_parquet(f"{FOLDER_PATH}/data/processed/outlier_removed.parquet")
        except:
            df = pd.read_parquet("https://storage.googleapis.com/wind_power_forecast/data/processed/outlier_removed.parquet")
        self.raw_df = df
        self.weather_cols = [col for col in df.columns if col.startswith(("UGRD", "VGRD"))]
        self.df = df[df["rt_plant_id"].isin(self.plants)]

    def _add_lagged(self, add_lagged):
        if add_lagged:
            self.df["production_48_lagged"] = self.df.groupby("rt_plant_id").production.shift(48)
            self.df = self.df.dropna()
            self.id_cols.append("production_48_lagged")

    def _add_speed(self, add_speed):
        if add_speed:
            df = self.df
            for box in ["SW", "NW", "NE", "SE"]:
                df[f"speed_{box}"] = np.sqrt(np.square(df[f"UGRD_80.m.above.ground.{box}"]) + np.square(df[f"VGRD_80.m.above.ground.{box}"]))
                df[f"angle_{box}"] = np.arctan(df[f"UGRD_80.m.above.ground.{box}"] / df[f"VGRD_80.m.above.ground.{box}"])
            self.df = df

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

def download_artifact(artifact):
    import wandb, os, json
    # 'merts/keras/run-3xa1vds1-valid_predictions:v0'
    artifact = wandb.use_artifact(artifact, type='run_table')
    artifact_dir = artifact.download()

    for file in os.listdir(artifact_dir):
        if file.endswith(".json"):
            with open(os.path.join(artifact_dir, file)) as json_data:
                data = json.load(json_data)
    df = pd.DataFrame(data["data"], columns=data["columns"])
    df["forecast_dt"] = pd.to_datetime(df["forecast_dt"])
    return df