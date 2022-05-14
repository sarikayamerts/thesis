import os
import numpy as np
import mxnet as mx
from tqdm import tqdm
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions

from .base_helpers import calculate_wmape

mx.random.seed(7)
np.random.seed(7)

CTX = "gpu" if mx.context.num_gpus() else "cpu" # add this to trainer
FREQ = "1H"
PREDICTION_LENGTH = int(os.environ.get("PREDICTION_LENGTH", 48))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", 72))


from dataclasses import dataclass

@dataclass
class GluonTSWrapper:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    prediction_length: int = 48
    context_length: int = 48

    def __post_init__(self):
        self.train_indices = sorted(self.train_df.index.unique())
        self.valid_indices = sorted(self.valid_df.index.unique())
        self.test_indices = sorted(self.test_df.index.unique())
        self.plants = sorted(self.train_df.rt_plant_id.unique())
        self.feature_columns = [col for col in self.train_df.columns if col not in ["production", "rt_plant_id"]]

        self.gluon_df = self.reduce_mem_usage(
            pd.concat([self.train_df, self.valid_df, self.test_df]))

        # train_ds, valid_ds, test_ds = self.prepare_gluon_datasets()
        # deepar_estimator = GluonTSModel(DeepAREstimator, params={"num_layers": 3})()
        # valid_predictions, test_predictions = self.make_prediction(deepar_estimator, valid_ds, test_ds)
        # valid_output, test_output = evaluate_prediction(valid_df, test_df, valid_predictions, test_predictions)
    @staticmethod
    def reduce_mem_usage(df):
        import gc; gc.collect()
        print(df.memory_usage(deep=True).sum() / 1024 ** 2)
        df["rt_plant_id"] = df["rt_plant_id"].astype(np.int16)
        for col in df.columns:
            if col not in ["rt_plant_id"]:
                df[col] = df[col].astype(np.float16)
        gc.collect()
        print(df.memory_usage(deep=True).sum() / 1024 ** 2)
        return df

    def prepare_gluon_datasets(self):
        train_data = self.gluon_df[self.gluon_df.index <= self.train_indices[-1]]
        train_ds = ListDataset([self.dataset_helper(train_data, plant_id=plant_id)
                                for plant_id in tqdm(self.gluon_df.rt_plant_id.unique())], freq=FREQ)

        valid_data = lambda date_shift: self.gluon_df[self.gluon_df.index < self.valid_indices[date_shift]]
        valid_ds = ListDataset([
            self.dataset_helper(valid_data(date_shift), plant_id=plant_id, is_train=False)
            for plant_id in tqdm(self.gluon_df.rt_plant_id.unique())
            for date_shift in [(i+1)*24 for i in range(len(self.valid_indices) // 24 - 1)]
        ], freq=FREQ)

        test_data = lambda date_shift: self.gluon_df[self.gluon_df.index < self.test_indices[date_shift]]
        test_ds = ListDataset([
            self.dataset_helper(test_data(date_shift), plant_id=plant_id, is_train=False)
            for plant_id in tqdm(self.gluon_df.rt_plant_id.unique())
            for date_shift in [(i+1)*24 for i in range(len(self.test_indices) // 24 - 1)]
        ], freq=FREQ)
        return train_ds, valid_ds, test_ds

    def dataset_helper(self, df_, plant_id, is_train=True):
        df__ = df_[df_["rt_plant_id"] == plant_id]
        if not is_train:
            # it's enough to look for only context + prediction length period
            df__ = df__.iloc[-(self.context_length+self.prediction_length):]
        return {
            "target": df__.production,
            "start": df__.index[0],
            "item_id": str(plant_id),
            "feat_dynamic_real": df__[self.feature_columns].T
        }

    def make_prediction(self, predictor, valid_ds, test_ds):
        valid_forecast_it, _ = make_evaluation_predictions(valid_ds, predictor=predictor, num_samples=100)
        valid_predictions = {k: [] for k in self.plants}
        for forecast_entry in valid_forecast_it:
            valid_predictions[int(forecast_entry.item_id)].append(forecast_entry.mean_ts[-24:])

        test_forecast_it, _ = make_evaluation_predictions(test_ds, predictor=predictor, num_samples=100)
        test_predictions = {k: [] for k in self.plants}
        for forecast_entry in test_forecast_it:
            test_predictions[int(forecast_entry.item_id)].append(forecast_entry.mean_ts[-24:])

        valid_predictions = self._prepare_output(valid_predictions)
        test_predictions = self._prepare_output(test_predictions)
        return valid_predictions, test_predictions

    def _prepare_output(self, predictions):
        return pd.concat([pd.concat(predictions[plant_id]).to_frame("prediction").assign(rt_plant_id=plant_id)
                          for plant_id in self.plants]).rename_axis("forecast_dt")

    @staticmethod
    def evaluate_prediction(valid_df, test_df, valid_predictions, test_predictions):
        valid_output = pd.merge(valid_df[["rt_plant_id", "production"]], valid_predictions, on=["forecast_dt", "rt_plant_id"])
        test_output = pd.merge(test_df[["rt_plant_id", "production"]], test_predictions, on=["forecast_dt", "rt_plant_id"])
        print("Total validation WMAPE:", calculate_wmape(valid_output["prediction"], valid_output["production"]))
        print(valid_output.groupby("rt_plant_id").apply(lambda x: calculate_wmape(x["prediction"], x["production"])).sort_values().to_frame("WMAPE").reset_index())

        print("Total test WMAPE:", calculate_wmape(test_output["prediction"], test_output["production"]))
        print(test_output.groupby("rt_plant_id").apply(lambda x: calculate_wmape(x["prediction"], x["production"])).sort_values().to_frame("WMAPE").reset_index())
        return valid_output, test_output

class GluonTSModel:
    def __init__(self, estimator, params):
        default_params = {
            "freq": FREQ,
            "prediction_length": PREDICTION_LENGTH,
            "context_length": CONTEXT_LENGTH,
            "use_feat_dynamic_real": True,
            "scaling": False
        }
        self.estimator = estimator
        self.params = default_params | params

    def __call__(self):
        return self.estimator(**self.params)


